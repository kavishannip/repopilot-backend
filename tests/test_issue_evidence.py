import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.summarizer import (
    RepoSummarizer,
    _analyze_dependency_usage,
    _build_positive_tech_evidence,
)


def test_dependency_usage_scans_all_selected_chunks_for_mongodb_imports():
    parsed_manifests = {
        "package.json": {
            "all_dependencies": {
                "mongodb": "^6.0.0",
            }
        }
    }
    full_file_contents = {
        "README.md": "This project uses MongoDB.",
    }
    all_chunks = [
        {
            "text": "import { MongoClient } from 'mongodb'\nexport const client = new MongoClient(uri)",
            "metadata": {"path": "src/lib/db.ts", "imports": ["import { MongoClient } from 'mongodb'"]},
        }
    ]

    result = _analyze_dependency_usage(parsed_manifests, full_file_contents, {}, all_chunks)

    assert result["mongodb"]["status"] == "observed_used"
    assert "mongodb" in result["mongodb"]["matched_imports"]
    assert "src/lib/db.ts" in result["mongodb"]["evidence"]


def test_positive_tech_evidence_detects_mongoose_usage_outside_search_results():
    all_chunks = [
        {
            "text": "import mongoose from 'mongoose'\nawait mongoose.connect(process.env.MONGODB_URI)",
            "metadata": {"path": "server/database.ts", "start_line": 4, "imports": ["import mongoose from 'mongoose'"]},
        }
    ]

    evidence = _build_positive_tech_evidence(all_chunks, {})

    assert "mongodb" in evidence
    assert any("server/database.ts" in ref for ref in evidence["mongodb"])


def test_flagged_issue_filter_removes_absence_claim_contradicted_by_mongodb_evidence():
    summarizer = object.__new__(RepoSummarizer)
    issues = [
        "README says MongoDB is used, but code does not use MongoDB.",
        "The API key is hard-coded in src/config.ts.",
    ]
    dep_usage = {
        "mongoose": {
            "status": "observed_used",
            "evidence": ["server/database.ts"],
        }
    }
    tech_evidence = {
        "mongodb": ["server/database.ts:L4: Mongoose connection call"],
    }

    filtered = summarizer._filter_unsupported_flagged_issues(issues, dep_usage, tech_evidence)

    assert filtered == ["The API key is hard-coded in src/config.ts."]


def test_flagged_issue_filter_removes_low_confidence_unused_dependency_claims():
    summarizer = object.__new__(RepoSummarizer)
    issues = [
        "The package mongodb appears unused because it was not found in code.",
    ]

    filtered = summarizer._filter_unsupported_flagged_issues(issues, {}, {})

    assert filtered == []
