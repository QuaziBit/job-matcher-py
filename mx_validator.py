"""
email_validator.py — Server-side email domain MX record validation.

Uses subprocess nslookup (available on Windows, macOS, Linux) to check
whether a domain has MX records. No third-party dependencies required.

Results are cached in the domain_mx_cache SQLite table (24-hour TTL).
"""

import asyncio
import logging
from analyzer.llm import _verbose
import re
import subprocess
from datetime import datetime, timedelta

import aiosqlite

logger = logging.getLogger("mx_validator")




# ── MX lookup via nslookup ────────────────────────────────────────────────────

def _nslookup_mx(domain: str, timeout: int = 5) -> list[str]:
    """
    Run nslookup -type=MX <domain> and parse out MX record hostnames.
    Returns a list of MX hostnames, empty list if none found or on error.
    Works on Windows (nslookup built-in), macOS, and Linux.
    """
    try:
        result = subprocess.run(
            ["nslookup", "-type=MX", domain],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout + result.stderr
        if _verbose():
            logger.info(f"→ nslookup output for {domain}:\n{output.strip()}")
        # Windows: "domain  MX preference = 10, mail exchanger = mx1.example.com"
        windows_hosts = re.findall(
            r"MX preference\s*=\s*\d+,\s*mail exchanger\s*=\s*(\S+)",
            output, re.IGNORECASE,
        )
        # Linux: "mail exchanger = 10 mx1.example.com"
        linux_hosts = re.findall(
            r"mail exchanger\s*=\s*\d+\s+(\S+)",
            output, re.IGNORECASE,
        )
        mx_hosts = windows_hosts + linux_hosts
        return [h.rstrip(".,").rstrip(".") for h in mx_hosts if h]
    except FileNotFoundError:
        logger.warning("nslookup not found — MX check unavailable")
        return []
    except subprocess.TimeoutExpired:
        logger.warning(f"nslookup timeout for domain: {domain}")
        return []
    except Exception as e:
        logger.warning(f"nslookup error for {domain}: {e}")
        return []


async def lookup_mx(domain: str) -> list[str]:
    """
    Async wrapper around _nslookup_mx — runs in a thread pool so it
    doesn't block the event loop.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _nslookup_mx, domain)


# ── Domain extraction ─────────────────────────────────────────────────────────

def extract_domain(email: str) -> str | None:
    """Extract domain from an email address. Returns None if invalid."""
    email = (email or "").strip().lower()
    if "@" not in email:
        return None
    parts = email.rsplit("@", 1)
    domain = parts[1].strip()
    if not domain or "." not in domain:
        return None
    return domain


# ── DB cache helpers ──────────────────────────────────────────────────────────

CACHE_TTL_HOURS = 24


async def get_cached_mx(db: aiosqlite.Connection, domain: str) -> dict | None:
    """
    Return cached MX result if it exists and is within TTL.
    Returns dict with {domain, has_mx, mx_records, checked_at} or None.
    """
    cutoff = (datetime.utcnow() - timedelta(hours=CACHE_TTL_HOURS)).isoformat()
    async with db.execute(
        """SELECT domain, has_mx, mx_records, checked_at
           FROM domain_mx_cache
           WHERE domain = ? AND checked_at > ?""",
        (domain, cutoff),
    ) as cur:
        row = await cur.fetchone()
    if not row:
        return None
    return {
        "domain":     row["domain"],
        "has_mx":     bool(row["has_mx"]),
        "mx_records": (row["mx_records"] or "").split(",") if row["mx_records"] else [],
        "checked_at": row["checked_at"],
        "cached":     True,
    }


async def upsert_mx_cache(
    db: aiosqlite.Connection,
    domain: str,
    has_mx: bool,
    mx_records: list[str],
) -> None:
    """Store or update MX lookup result in cache."""
    mx_str = ",".join(mx_records) if mx_records else ""
    await db.execute(
        """INSERT INTO domain_mx_cache (domain, has_mx, mx_records, checked_at)
           VALUES (?, ?, ?, CURRENT_TIMESTAMP)
           ON CONFLICT(domain) DO UPDATE SET
               has_mx      = excluded.has_mx,
               mx_records  = excluded.mx_records,
               checked_at  = excluded.checked_at""",
        (domain, int(has_mx), mx_str),
    )
    await db.commit()


# ── Main validation entry point ───────────────────────────────────────────────

async def validate_email_domain(
    email: str,
    db: aiosqlite.Connection,
) -> dict:
    """
    Validate the MX records for an email address domain.

    Returns:
        {
            "email":      str,
            "domain":     str | None,
            "valid":      bool,   # False if no domain or no MX records
            "has_mx":     bool,
            "mx_records": list[str],
            "cached":     bool,
            "error":      str | None,
        }
    """
    domain = extract_domain(email)
    if not domain:
        return {
            "email":      email,
            "domain":     None,
            "valid":      False,
            "has_mx":     False,
            "mx_records": [],
            "cached":     False,
            "error":      "Invalid email address — no domain found",
        }

    # Check cache first
    cached = await get_cached_mx(db, domain)
    if cached:
        logger.info(f"→ MX cache hit for {domain}: has_mx={cached['has_mx']}")
        return {
            "email":      email,
            "domain":     domain,
            "valid":      cached["has_mx"],
            "has_mx":     cached["has_mx"],
            "mx_records": cached["mx_records"],
            "cached":     True,
            "error":      None,
        }

    # Live lookup
    logger.info(f"→ MX lookup for {domain}")
    mx_records = await lookup_mx(domain)
    has_mx = len(mx_records) > 0
    logger.info(f"→ MX result for {domain}: has_mx={has_mx} records={mx_records}")

    # Cache result
    await upsert_mx_cache(db, domain, has_mx, mx_records)

    return {
        "email":      email,
        "domain":     domain,
        "valid":      has_mx,
        "has_mx":     has_mx,
        "mx_records": mx_records,
        "cached":     False,
        "error":      None,
    }
