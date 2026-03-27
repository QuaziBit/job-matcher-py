# skills.py — Skill normalization and category mapping

SKILL_ALIASES = {
    "js":             "JavaScript",
    "javascript":     "JavaScript",
    "ts":             "TypeScript",
    "typescript":     "TypeScript",
    "postgres":       "PostgreSQL",
    "postgresql":     "PostgreSQL",
    "psql":           "PostgreSQL",
    "k8s":            "Kubernetes",
    "kubernetes":     "Kubernetes",
    "rest api":       "REST APIs",
    "rest apis":      "REST APIs",
    "restful":        "REST APIs",
    "restful api":    "REST APIs",
    "ci/cd":          "CI/CD",
    "cicd":           "CI/CD",
    "node":           "Node.js",
    "node.js":        "Node.js",
    "nodejs":         "Node.js",
    "react.js":       "React",
    "reactjs":        "React",
    "vue.js":         "Vue",
    "vuejs":          "Vue",
    "ml":             "Machine Learning",
    "ai/ml":          "AI/ML",
    "llms":           "LLMs",
    "large language": "LLMs",
    "gcp":            "Google Cloud",
    "aws":            "AWS",
    "amazon web":     "AWS",
    "azure":          "Azure",
    "docker":         "Docker",
    "containers":     "Docker",
    "terraform":      "Terraform",
    "iac":            "Infrastructure as Code",
    "mongo":          "MongoDB",
    "mongodb":        "MongoDB",
    "redis":          "Redis",
    "elasticsearch":  "Elasticsearch",
    "elastic":        "Elasticsearch",
    "splunk":         "Splunk",
    "security+":      "CompTIA Security+",
    "sec+":           "CompTIA Security+",
}

SKILL_CATEGORIES = {
    "JavaScript":        "frontend",
    "TypeScript":        "frontend",
    "React":             "frontend",
    "Angular":           "frontend",
    "Vue":               "frontend",
    "HTML":              "frontend",
    "CSS":               "frontend",
    "Python":            "backend",
    "Go":                "backend",
    "Java":              "backend",
    "Node.js":           "backend",
    "Flask":             "backend",
    "FastAPI":           "backend",
    "Spring Boot":       "backend",
    "REST APIs":         "backend",
    "PostgreSQL":        "database",
    "MySQL":             "database",
    "SQLite":            "database",
    "MongoDB":           "database",
    "Redis":             "database",
    "Elasticsearch":     "database",
    "AWS":               "cloud",
    "Azure":             "cloud",
    "Google Cloud":      "cloud",
    "Docker":            "devops",
    "Kubernetes":        "devops",
    "CI/CD":             "devops",
    "Jenkins":           "devops",
    "Terraform":         "devops",
    "Splunk":            "security",
    "CompTIA Security+": "security",
    "IAM":               "security",
    "Zero Trust":        "security",
    "LLMs":              "ai",
    "Machine Learning":  "ai",
    "Anthropic":         "ai",
    "Ollama":            "ai",
    "RAG":               "ai",
}


def normalize_skill(skill: str) -> str:
    """Return the canonical form of a skill name."""
    lower = skill.strip().lower()
    return SKILL_ALIASES.get(lower, skill.strip())


def get_skill_category(skill: str) -> str:
    """Return the category for a skill, or 'other' if unknown."""
    normalized = normalize_skill(skill)
    return SKILL_CATEGORIES.get(normalized, "other")


def cluster_penalty_cap(group: str) -> int:
    """Return the maximum total penalty allowed for a skill cluster group."""
    if group == "security":
        return 2
    return 1
