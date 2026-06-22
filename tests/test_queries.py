"""
tests/test_queries.py — Tests for queries.py, a module of SQL string
constants and one builder function (no async functions, no DB connection
handling — see queries.py's module docstring).

These tests check that each constant:
  - exists and is a non-empty string (or list, for MIGRATIONS)
  - references the expected table and columns
  - has the right shape (placeholder count matches parameter count used
    at the call site, format() slots are present where expected)

They do NOT execute the SQL against a real database — that's already
covered by tests/test_api.py, which exercises every route end-to-end
against a real temp DB and would fail if any of these queries were wrong.
This file catches typos, accidental table/column renames, and structural
mistakes (e.g. wrong placeholder count) without needing a DB connection.
"""

import re
import unittest

import queries as q


class TestJobsQueries(unittest.TestCase):
    """Each jobs-table constant references the jobs table and the right
    columns, with the placeholder count callers expect."""

    def test_get_job_id_by_url(self):
        self.assertIn("FROM jobs", q.GET_JOB_ID_BY_URL)
        self.assertIn("WHERE url = ?", q.GET_JOB_ID_BY_URL)
        self.assertEqual(q.GET_JOB_ID_BY_URL.count("?"), 1)

    def test_get_manual_job_id_by_url_or_slug(self):
        self.assertIn("FROM jobs", q.GET_MANUAL_JOB_ID_BY_URL_OR_SLUG)
        # Called with (job_url, synthetic_url, source_url) — 3 params
        self.assertEqual(q.GET_MANUAL_JOB_ID_BY_URL_OR_SLUG.count("?"), 3)

    def test_insert_job_with_company_url(self):
        self.assertIn("INSERT INTO jobs", q.INSERT_JOB_WITH_COMPANY_URL)
        self.assertIn("company_url", q.INSERT_JOB_WITH_COMPANY_URL)
        # url, title, company, location, company_url, raw_description
        self.assertEqual(q.INSERT_JOB_WITH_COMPANY_URL.count("?"), 6)

    def test_insert_job_no_company_url(self):
        self.assertIn("INSERT INTO jobs", q.INSERT_JOB_NO_COMPANY_URL)
        self.assertNotIn("company_url", q.INSERT_JOB_NO_COMPANY_URL)
        # url, title, company, location, raw_description
        self.assertEqual(q.INSERT_JOB_NO_COMPANY_URL.count("?"), 5)

    def test_get_job_raw_description(self):
        self.assertIn("raw_description", q.GET_JOB_RAW_DESCRIPTION)
        self.assertEqual(q.GET_JOB_RAW_DESCRIPTION.count("?"), 1)

    def test_get_job_for_salary(self):
        for col in ("title", "company", "location", "raw_description", "salary_estimate"):
            self.assertIn(col, q.GET_JOB_FOR_SALARY)
        self.assertEqual(q.GET_JOB_FOR_SALARY.count("?"), 1)

    def test_update_job_salary_estimate(self):
        self.assertIn("UPDATE jobs SET salary_estimate", q.UPDATE_JOB_SALARY_ESTIMATE)
        self.assertEqual(q.UPDATE_JOB_SALARY_ESTIMATE.count("?"), 2)

    def test_clear_job_salary_estimate(self):
        self.assertIn("salary_estimate = ''", q.CLEAR_JOB_SALARY_ESTIMATE)
        self.assertEqual(q.CLEAR_JOB_SALARY_ESTIMATE.count("?"), 1)

    def test_delete_job(self):
        self.assertEqual(q.DELETE_JOB.strip(), "DELETE FROM jobs WHERE id = ?")

    def test_get_job_id_and_description(self):
        self.assertIn("id, raw_description", q.GET_JOB_ID_AND_DESCRIPTION)

    def test_get_job_id_only(self):
        self.assertEqual(q.GET_JOB_ID_ONLY.strip(), "SELECT id FROM jobs WHERE id = ?")

    def test_get_job_id_and_company(self):
        self.assertIn("id, company", q.GET_JOB_ID_AND_COMPANY)

    def test_update_job_url(self):
        self.assertIn("SET url = ?", q.UPDATE_JOB_URL)

    def test_update_job_title(self):
        self.assertIn("SET title = ?", q.UPDATE_JOB_TITLE)

    def test_update_job_company(self):
        self.assertIn("SET company = ?", q.UPDATE_JOB_COMPANY)

    def test_update_job_location(self):
        self.assertIn("SET location = ?", q.UPDATE_JOB_LOCATION)

    def test_update_job_company_url(self):
        self.assertIn("SET company_url = ?", q.UPDATE_JOB_COMPANY_URL)

    def test_get_job_full_row(self):
        self.assertEqual(q.GET_JOB_FULL_ROW.strip(), "SELECT * FROM jobs WHERE id = ?")

    def test_count_jobs_has_where_sql_slot(self):
        self.assertIn("{where_sql}", q.COUNT_JOBS)
        self.assertIn("COUNT(*)", q.COUNT_JOBS)
        # format() with where_sql="" should still be valid-looking SQL
        formatted = q.COUNT_JOBS.format(where_sql="")
        self.assertNotIn("{", formatted)
        formatted_with_filter = q.COUNT_JOBS.format(where_sql="WHERE j.company = ?")
        self.assertIn("WHERE j.company = ?", formatted_with_filter)

    def test_list_jobs_paginated_has_where_sql_slot(self):
        self.assertIn("{where_sql}", q.LIST_JOBS_PAGINATED)
        for col in ("best_score", "adjusted_score", "provider", "last_model", "has_recruiter"):
            self.assertIn(col, q.LIST_JOBS_PAGINATED)
        formatted = q.LIST_JOBS_PAGINATED.format(where_sql="")
        self.assertNotIn("{", formatted)

    def test_get_vetting_rows(self):
        self.assertIn("FROM jobs j", q.GET_VETTING_ROWS)
        self.assertIn("LEFT JOIN applications", q.GET_VETTING_ROWS)
        self.assertIn("COLLATE NOCASE", q.GET_VETTING_ROWS)
        # No parameters — this query takes none
        self.assertEqual(q.GET_VETTING_ROWS.count("?"), 0)


class TestCompanyMetaQueries(unittest.TestCase):

    def test_get_company_meta(self):
        self.assertEqual(q.GET_COMPANY_META.strip(), "SELECT * FROM company_meta WHERE company_name = ?")

    def test_insert_or_ignore_company_meta(self):
        self.assertIn("INSERT OR IGNORE INTO company_meta", q.INSERT_OR_IGNORE_COMPANY_META)
        self.assertEqual(q.INSERT_OR_IGNORE_COMPANY_META.count("?"), 1)

    def test_upsert_company_meta_has_format_slots(self):
        for slot in ("{cols}", "{placeholders}", "{set_clause}"):
            self.assertIn(slot, q.UPSERT_COMPANY_META)
        formatted = q.UPSERT_COMPANY_META.format(
            cols="company_name, bbb_rating", placeholders="?, ?",
            set_clause="bbb_rating = excluded.bbb_rating",
        )
        self.assertNotIn("{", formatted)
        self.assertIn("ON CONFLICT(company_name)", formatted)

    def test_upsert_company_vetting(self):
        for col in ("llm_risk_level", "llm_assessment", "llm_signals", "llm_provider", "llm_model", "llm_assessed_at"):
            self.assertIn(col, q.UPSERT_COMPANY_VETTING)
        # company_name, risk_level, assessment, signals, provider, model
        self.assertEqual(q.UPSERT_COMPANY_VETTING.count("?"), 6)

    def test_sync_company_url_to_meta(self):
        self.assertIn("INSERT INTO company_meta (company_name, company_url)", q.SYNC_COMPANY_URL_TO_META)
        self.assertIn("ON CONFLICT(company_name)", q.SYNC_COMPANY_URL_TO_META)
        self.assertEqual(q.SYNC_COMPANY_URL_TO_META.count("?"), 2)

    def test_delete_company_meta(self):
        self.assertEqual(q.DELETE_COMPANY_META.strip(), "DELETE FROM company_meta WHERE company_name = ?")

    def test_rename_company_meta(self):
        self.assertIn("UPDATE company_meta SET company_name = ?", q.RENAME_COMPANY_META)
        self.assertEqual(q.RENAME_COMPANY_META.count("?"), 2)

    def test_get_company_meta_batch_query_builds_correct_placeholder_count(self):
        for n in (1, 2, 5, 10):
            query = q.get_company_meta_batch_query(n)
            self.assertEqual(query.count("?"), n, f"expected {n} placeholders for n={n}")

    def test_get_company_meta_batch_query_zero_companies(self):
        query = q.get_company_meta_batch_query(0)
        self.assertEqual(query.count("?"), 0)
        self.assertIn("IN ()", query)

    def test_get_company_meta_batch_query_references_expected_columns(self):
        query = q.get_company_meta_batch_query(3)
        for col in ("company_name", "company_url", "glassdoor_rating", "llm_risk_level"):
            self.assertIn(col, query)

    def test_get_company_meta_batch_query_no_leftover_format_braces(self):
        query = q.get_company_meta_batch_query(4)
        self.assertNotIn("{", query)
        self.assertNotIn("}", query)


class TestAnalysesQueries(unittest.TestCase):

    def test_insert_analysis_column_and_placeholder_count_match(self):
        # Extract the column list between the parens after "analyses"
        match = re.search(r"INSERT INTO analyses\s*\(([^)]+)\)", q.INSERT_ANALYSIS)
        self.assertIsNotNone(match)
        columns = [c.strip() for c in match.group(1).split(",")]
        self.assertEqual(len(columns), q.INSERT_ANALYSIS.count("?"),
                          "column count must match placeholder count")

    def test_insert_analysis_has_expected_columns(self):
        for col in ("job_id", "resume_id", "score", "adjusted_score", "analysis_mode"):
            self.assertIn(col, q.INSERT_ANALYSIS)

    def test_get_analysis_id(self):
        self.assertEqual(q.GET_ANALYSIS_ID.strip(), "SELECT id FROM analyses WHERE id = ?")

    def test_delete_analysis(self):
        self.assertEqual(q.DELETE_ANALYSIS.strip(), "DELETE FROM analyses WHERE id = ?")

    def test_get_analyses_with_resume_label(self):
        self.assertIn("JOIN resumes", q.GET_ANALYSES_WITH_RESUME_LABEL)
        self.assertIn("resume_label", q.GET_ANALYSES_WITH_RESUME_LABEL)
        self.assertIn("ORDER BY a.created_at DESC", q.GET_ANALYSES_WITH_RESUME_LABEL)
        self.assertEqual(q.GET_ANALYSES_WITH_RESUME_LABEL.count("?"), 1)


class TestApplicationsQueries(unittest.TestCase):

    def test_upsert_application_column_and_placeholder_count_match(self):
        match = re.search(r"INSERT INTO applications\s*\(([^)]+)\)", q.UPSERT_APPLICATION)
        self.assertIsNotNone(match)
        columns = [c.strip() for c in match.group(1).split(",")]
        # Only the INSERT clause's placeholders should match the column count
        # (the ON CONFLICT...excluded clause has no extra ? placeholders)
        self.assertEqual(len(columns), q.UPSERT_APPLICATION.count("?"))

    def test_upsert_application_has_on_conflict(self):
        self.assertIn("ON CONFLICT(job_id)", q.UPSERT_APPLICATION)
        self.assertIn("updated_at=CURRENT_TIMESTAMP", q.UPSERT_APPLICATION)

    def test_get_application_by_job(self):
        self.assertEqual(q.GET_APPLICATION_BY_JOB.strip(), "SELECT * FROM applications WHERE job_id = ?")


class TestJobEmailsQueries(unittest.TestCase):

    def test_get_job_email(self):
        for col in ("id", "raw_html", "created_at"):
            self.assertIn(col, q.GET_JOB_EMAIL)
        self.assertEqual(q.GET_JOB_EMAIL.count("?"), 1)

    def test_upsert_job_email(self):
        self.assertIn("INSERT INTO job_emails", q.UPSERT_JOB_EMAIL)
        self.assertIn("ON CONFLICT(job_id)", q.UPSERT_JOB_EMAIL)
        self.assertEqual(q.UPSERT_JOB_EMAIL.count("?"), 2)

    def test_delete_job_email(self):
        self.assertEqual(q.DELETE_JOB_EMAIL.strip(), "DELETE FROM job_emails WHERE job_id = ?")


class TestResumesQueries(unittest.TestCase):

    def test_insert_resume(self):
        self.assertIn("INSERT INTO resumes (label, content)", q.INSERT_RESUME)
        self.assertEqual(q.INSERT_RESUME.count("?"), 2)

    def test_get_resume(self):
        for col in ("label", "content", "created_at", "char_count"):
            self.assertIn(col, q.GET_RESUME)
        self.assertEqual(q.GET_RESUME.count("?"), 1)

    def test_get_resume_content(self):
        self.assertEqual(q.GET_RESUME_CONTENT.strip(), "SELECT content FROM resumes WHERE id = ?")

    def test_delete_resume(self):
        self.assertEqual(q.DELETE_RESUME.strip(), "DELETE FROM resumes WHERE id = ?")

    def test_list_resume_ids_and_labels(self):
        self.assertIn("SELECT id, label FROM resumes", q.LIST_RESUME_IDS_AND_LABELS)
        self.assertIn("ORDER BY created_at DESC", q.LIST_RESUME_IDS_AND_LABELS)
        self.assertEqual(q.LIST_RESUME_IDS_AND_LABELS.count("?"), 0)

    def test_list_resumes_full(self):
        for col in ("label", "created_at", "char_count"):
            self.assertIn(col, q.LIST_RESUMES_FULL)
        self.assertEqual(q.LIST_RESUMES_FULL.count("?"), 0)


class TestMxCacheQueries(unittest.TestCase):

    def test_get_mx_cache(self):
        for col in ("domain", "has_mx", "mx_records"):
            self.assertIn(col, q.GET_MX_CACHE)
        self.assertEqual(q.GET_MX_CACHE.count("?"), 0)


class TestSchemaAndMigrations(unittest.TestCase):
    """Sanity checks for the schema script and migrations list — these are
    exercised for real (against an actual SQLite DB) by every test in
    test_api.py via database.init_db(), but it's still worth asserting the
    raw text itself looks structurally correct."""

    EXPECTED_TABLES = (
        "resumes", "jobs", "analyses", "applications",
        "job_emails", "company_meta", "domain_mx_cache",
    )

    def test_schema_creates_all_expected_tables(self):
        for table in self.EXPECTED_TABLES:
            self.assertIn(
                f"CREATE TABLE IF NOT EXISTS {table}", q.SCHEMA,
                f"SCHEMA is missing table: {table}",
            )

    def test_schema_table_count_matches_expected(self):
        found = re.findall(r"CREATE TABLE IF NOT EXISTS (\w+)", q.SCHEMA)
        self.assertEqual(sorted(found), sorted(self.EXPECTED_TABLES))

    def test_schema_uses_if_not_exists_everywhere(self):
        # Every CREATE TABLE statement must be idempotent — running SCHEMA
        # against an existing DB (e.g. on every app start) must not error.
        create_count = q.SCHEMA.count("CREATE TABLE")
        if_not_exists_count = q.SCHEMA.count("CREATE TABLE IF NOT EXISTS")
        self.assertEqual(create_count, if_not_exists_count)

    def test_migrations_is_nonempty_list(self):
        self.assertIsInstance(q.MIGRATIONS, list)
        self.assertGreater(len(q.MIGRATIONS), 0)

    def test_migrations_are_all_alter_table_statements(self):
        for migration in q.MIGRATIONS:
            self.assertTrue(
                migration.strip().upper().startswith("ALTER TABLE"),
                f"non-ALTER-TABLE entry in MIGRATIONS: {migration!r}",
            )

    def test_migrations_only_target_known_tables(self):
        for migration in q.MIGRATIONS:
            match = re.match(r"ALTER TABLE (\w+)", migration.strip())
            self.assertIsNotNone(match, f"could not parse table name from: {migration!r}")
            self.assertIn(
                match.group(1), self.EXPECTED_TABLES,
                f"migration targets a table not in SCHEMA: {migration!r}",
            )

    def test_migrations_have_no_duplicate_column_adds(self):
        """Adding the same column twice across migrations would be a typo
        (the second ALTER would always fail silently due to the try/except
        in init_db, masking the mistake) — assert each (table, column) pair
        appears at most once."""
        seen = set()
        for migration in q.MIGRATIONS:
            match = re.match(r"ALTER TABLE (\w+) ADD COLUMN (\w+)", migration.strip())
            self.assertIsNotNone(match, f"unexpected migration format: {migration!r}")
            key = (match.group(1), match.group(2))
            self.assertNotIn(key, seen, f"duplicate migration for {key}")
            seen.add(key)


if __name__ == "__main__":
    unittest.main()
