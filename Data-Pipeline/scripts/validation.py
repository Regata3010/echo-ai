# # Data-Pipeline/scripts/validation.py
# import pandas as pd
# import numpy as np
# import json
# import logging
# from typing import Dict, Any
# import os

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# def check_data_types(df: pd.DataFrame) -> Dict[str, Any]:
#     """Validate data types"""
#     expected_types = {
#         'reviewRating': 'int64',
#         'reviewText': 'object',
#     }
    
#     type_check_results = {}
#     for col, expected_type in expected_types.items():
#         if col in df.columns:
#             actual_type = str(df[col].dtype)
#             type_check_results[col] = {
#                 'expected': expected_type,
#                 'actual': actual_type,
#                 'valid': expected_type in actual_type
#             }
    
#     return type_check_results

# def check_data_ranges(df: pd.DataFrame) -> Dict[str, Any]:
#     """Validate data ranges"""
#     range_checks = {}
    
#     # Check rating range
#     if 'rating' in df.columns:
#         range_checks['reviewRating'] = {
#             'min': df['reviewRating'].min(),
#             'max': df['reviewRating'].max(),
#             'valid': (df['reviewRating'].min() >= 1) and (df['reviewRating'].max() <= 5),
#             'out_of_range_count': len(df[(df['rating'] < 1) | (df['reviewRating'] > 5)])
#         }
    
#     # Check text length
#     if 'text_length' in df.columns:
#         range_checks['text_length'] = {
#             'min': df['text_length'].min(),
#             'max': df['text_length'].max(),
#             'mean': df['text_length'].mean(),
#             'empty_count': len(df[df['text_length'] == 0])
#         }
    
#     return range_checks

# def check_missing_values(df: pd.DataFrame) -> Dict[str, Any]:
#     """Check for missing values"""
#     missing_report = {}
    
#     for col in df.columns:
#         missing_count = df[col].isnull().sum()
#         missing_pct = (missing_count / len(df)) * 100
        
#         missing_report[col] = {
#             'missing_count': int(missing_count),
#             'missing_percentage': round(missing_pct, 2),
#             'is_critical': col in ['review_id', 'rating', 'text']
#         }
    
#     return missing_report

# def check_duplicates(df: pd.DataFrame) -> Dict[str, Any]:
#     """Check for duplicate records"""
#     duplicate_report = {
#         'duplicate_review_ids': len(df[df.duplicated(subset=['authorName'], keep=False)]),
#         'duplicate_texts': len(df[df.duplicated(subset=['reviewText'], keep=False)]),
#         'exact_duplicates': len(df[df.duplicated(keep=False)])
#     }
    
#     return duplicate_report

# def validate_data(input_path: str = '../../data/raw/featurecleaned_apify_processed.csv') -> Dict[str, Any]:
#     """Main validation pipeline"""
#     try:
#         # Load data
#         logger.info(f"Loading data for validation from {input_path}")
#         df = pd.read_csv(input_path)
        
#         # Perform validation checks
#         validation_results = {
#             'file_path': input_path,
#             'total_records': len(df),
#             'total_columns': len(df.columns),
#             'column_names': list(df.columns),
#             'data_types': check_data_types(df),
#             'data_ranges': check_data_ranges(df),
#             'missing_values': check_missing_values(df),
#             'duplicates': check_duplicates(df),
#             'validation_timestamp': pd.Timestamp.now().isoformat()
#         }
        
#         # Determine overall validation status
#         validation_passed = True
#         issues = []
        
#         # Check for critical issues
#         if validation_results['duplicates']['duplicate_review_ids'] > 0:
#             issues.append("Duplicate review IDs found")
#             validation_passed = False
        
#         for col, missing_info in validation_results['missing_values'].items():
#             if missing_info['is_critical'] and missing_info['missing_count'] > 0:
#                 issues.append(f"Critical column '{col}' has missing values")
#                 validation_passed = False
        
#         validation_results['validation_passed'] = validation_passed
#         validation_results['issues'] = issues
        
#         # Save validation results
#         os.makedirs('data/metrics', exist_ok=True)
#         output_path = 'data/metrics/validation_results_mac.json'
#         with open(output_path, 'w') as f:
#             json.dump(validation_results, f, indent=2, default=str)
        
#         logger.info(f"Validation complete. Passed: {validation_passed}")
#         logger.info(f"Results saved to {output_path}")
        
#         if issues:
#             logger.warning(f"Validation issues: {', '.join(issues)}")
        
#         return validation_results
        
#     except Exception as e:
#         logger.error(f"Validation failed: {e}")
#         raise

# if __name__ == "__main__":
#     validate_data()

# Data-Pipeline/scripts/validation.py

import pandas as pd
import numpy as np
import json
import logging
from typing import Dict, Any
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_data_types(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate data types"""
    expected_types = {
        "reviewRating": "int64",
        "reviewText": "object",
    }

    type_check_results: Dict[str, Any] = {}
    for col, expected_type in expected_types.items():
        if col in df.columns:
            actual_type = str(df[col].dtype)
            type_check_results[col] = {
                "expected": expected_type,
                "actual": actual_type,
                "valid": expected_type in actual_type,
            }

    return type_check_results


def check_data_ranges(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate data ranges"""
    range_checks: Dict[str, Any] = {}

    # Check rating range
    if "reviewRating" in df.columns:
        min_rating = df["reviewRating"].min()
        max_rating = df["reviewRating"].max()
        out_of_range_mask = (df["reviewRating"] < 1) | (df["reviewRating"] > 5)
        range_checks["reviewRating"] = {
            "min": float(min_rating) if pd.notnull(min_rating) else None,
            "max": float(max_rating) if pd.notnull(max_rating) else None,
            "valid": (min_rating >= 1) and (max_rating <= 5),
            "out_of_range_count": int(out_of_range_mask.sum()),
        }

    # Check text length (if feature_engineering has run)
    if "text_length" in df.columns:
        min_len = df["text_length"].min()
        max_len = df["text_length"].max()
        mean_len = df["text_length"].mean()
        empty_count = int((df["text_length"] == 0).sum())

        range_checks["text_length"] = {
            "min": float(min_len) if pd.notnull(min_len) else None,
            "max": float(max_len) if pd.notnull(max_len) else None,
            "mean": float(mean_len) if pd.notnull(mean_len) else None,
            "empty_count": empty_count,
        }

    return range_checks


def check_missing_values(df: pd.DataFrame) -> Dict[str, Any]:
    """Check for missing values"""
    missing_report: Dict[str, Any] = {}

    for col in df.columns:
        missing_count = int(df[col].isnull().sum())
        missing_pct = (missing_count / len(df)) * 100 if len(df) > 0 else 0.0

        # Treat these as critical if missing
        is_critical = col in ["reviewText", "reviewRating", "placeName"]

        missing_report[col] = {
            "missing_count": missing_count,
            "missing_percentage": round(missing_pct, 2),
            "is_critical": is_critical,
        }

    return missing_report


def check_duplicates(df: pd.DataFrame) -> Dict[str, Any]:
    """Check for duplicate records

    NOTE: This no longer treats repeated authorName as duplicate IDs.
    It only reports:
    - duplicate_texts: same reviewText
    - exact_duplicates: completely identical rows (critical)
    """
    # How many rows share the same reviewText
    if "reviewText" in df.columns:
        duplicate_texts = int(df.duplicated(subset=["reviewText"], keep=False).sum())
    else:
        duplicate_texts = 0

    # Exact duplicate rows
    exact_duplicates = int(df.duplicated(keep=False).sum())

    duplicate_report: Dict[str, Any] = {
        # Kept for backward compatibility, but now based on exact duplicates
        # so your downstream code still sees the key.
        "duplicate_review_ids": exact_duplicates,
        "duplicate_texts": duplicate_texts,
        "exact_duplicates": exact_duplicates,
    }

    return duplicate_report


def validate_data(
    input_path: str = "../../data/raw/featurecleaned_apify_processed.csv",
) -> Dict[str, Any]:
    """Main validation pipeline"""
    try:
        # Load data
        logger.info(f"Loading data for validation from {input_path}")
        df = pd.read_csv(input_path)

        # Perform validation checks
        validation_results: Dict[str, Any] = {
            "file_path": input_path,
            "total_records": int(len(df)),
            "total_columns": int(len(df.columns)),
            "column_names": list(df.columns),
            "data_types": check_data_types(df),
            "data_ranges": check_data_ranges(df),
            "missing_values": check_missing_values(df),
            "duplicates": check_duplicates(df),
            "validation_timestamp": pd.Timestamp.now().isoformat(),
        }

        # Determine overall validation status
        validation_passed = True
        issues = []

        # Critical duplicate issue: exact duplicate rows
        if validation_results["duplicates"]["exact_duplicates"] > 0:
            issues.append("Exact duplicate rows found")
            validation_passed = False

        # Critical missing values
        for col, missing_info in validation_results["missing_values"].items():
            if missing_info["is_critical"] and missing_info["missing_count"] > 0:
                issues.append(f"Critical column '{col}' has missing values")
                validation_passed = False

        validation_results["validation_passed"] = validation_passed
        validation_results["issues"] = issues

        # Save validation results
        os.makedirs("data/metrics", exist_ok=True)
        output_path = "data/metrics/validation_results_mac.json"

        with open(output_path, "w") as f:
            json.dump(validation_results, f, indent=2, default=str)

        logger.info(f"Validation complete. Passed: {validation_passed}")
        logger.info(f"Results saved to {output_path}")

        if issues:
            logger.warning(f"Validation issues: {', '.join(issues)}")

        return validation_results

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise


if __name__ == "__main__":
    validate_data()
