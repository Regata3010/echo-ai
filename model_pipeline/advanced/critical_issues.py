"""
Critical Issue Detection Module
Detects health violations, safety issues, and other urgent problems
"""
import logging
from typing import Dict, List
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CriticalIssueDetector:
    """
    Detects critical issues that require immediate escalation
    These override normal sentiment analysis
    """
    
    def __init__(self):
        # Critical issue categories with keywords
        self.issue_keywords = {
            'health_violation': {
                'keywords': [
                    'hair','bug','insect', 'roach', 'rat', 'mouse', 'mold', 'moldy',
                    'sick', 'food poisoning', 'vomit', 'diarrhea', 'illness',
                    'undercooked', 'raw', 'spoiled', 'rotten', 'expired',
                    'contaminated', 'unsanitary', 'health code', 'health department'
                ],
                'exclusions': ['grate', 'grateful', 'ingrate'],
                'severity': 'critical',
                'requires_action': 'immediate_manager_contact'
            },
            
            'safety': {
                'keywords': [
                    'unsafe', 'danger', 'hazard', 'injury', 'injured', 'hurt',
                    'broken glass', 'sharp', 'slippery', 'fell', 'tripped',
                    'fire hazard', 'blocked exit', 'no fire extinguisher'
                ],
                'severity': 'critical',
                'requires_action': 'immediate_manager_contact'
            },
            
            'discrimination': {
                'keywords': [
                    'racist', 'racism', 'discrimination', 'discriminated',
                    'refused service', 'kicked out', 'profiling',
                    'homophobic', 'sexist', 'ageist'
                ],
                'severity': 'critical',
                'requires_action': 'legal_review'
            },
            
            'theft_fraud': {
                'keywords': [
                    'stole', 'stolen', 'theft', 'charged twice', 'overcharged',
                    'scam', 'fraud', 'credit card', 'unauthorized charge'
                ],
                'severity': 'high',
                'requires_action': 'management_review'
            },
            
            'extreme_negative': {
                'keywords': [
                    'never coming back', 'will never', 'worst ever',
                    'absolutely horrible', 'complete disaster', 'total waste',
                    'avoid at all costs', 'do not go', 'stay away'
                ],
                'severity': 'high',
                'requires_action': 'priority_response'
            }
        }
    
    def detect(self, text: str) -> Dict:
        """Detect critical issues in review"""
        
        text_lower = text.lower()
        
        issues_found = []
        severity_level = 'normal'
        actions_required = []
        
        # Check each category
        for category, config in self.issue_keywords.items():
            matches = []
            exclusions = config.get('exclusions', [])
            
            for keyword in config['keywords']:
                if keyword in text_lower:
                    # Check exclusions
                    if not any(excl in text_lower for excl in exclusions):
                        matches.append(keyword)
            
            if matches:
                issues_found.append({
                    'category': category,
                    'matched_keywords': matches,
                    'severity': config['severity'],
                    'action': config['requires_action']
                })
                
                if config['severity'] == 'critical':
                    severity_level = 'critical'
                elif config['severity'] == 'high' and severity_level != 'critical':
                    severity_level = 'high'
                
                actions_required.append(config['requires_action'])
        
        has_critical = len(issues_found) > 0
        
        return {
            'has_critical_issue': has_critical,
            'severity': severity_level,
            'issues': issues_found,
            'actions_required': list(set(actions_required)),
            'should_escalate': severity_level == 'critical',
            'should_override_sentiment': has_critical
    }
    
    def get_escalation_message(self, issues: Dict) -> str:
        """
        Generate internal escalation message for critical issues
        """
        if not issues['has_critical_issue']:
            return None
        
        categories = [issue['category'] for issue in issues['issues']]
        
        messages = {
            'health_violation': "URGENT: Health violation reported. Immediate investigation required.",
            'safety': "URGENT: Safety concern reported. Immediate review and action needed.",
            'discrimination': "CRITICAL: Discrimination allegation. Legal review required immediately.",
            'theft_fraud': "HIGH PRIORITY: Fraud/theft allegation. Management review needed.",
            'extreme_negative': "HIGH PRIORITY: Extremely negative review. Priority response needed."
        }
        
        escalation_messages = [messages.get(cat, f"Issue: {cat}") for cat in categories]
        
        return " | ".join(escalation_messages)


def test_critical_detector():
    """Test critical issue detection"""
    
    detector = CriticalIssueDetector()
    
    test_cases = [
        "Found a hair in my salad. Otherwise food was decent.",
        "Great food and service!",
        "We got food poisoning from the chicken.",
        "Service was slow but acceptable.",
        "Absolutely horrible. Worst ever. Never coming back."
    ]
    
    print("Critical Issue Detection Test")
    print("="*60)
    
    for text in test_cases:
        result = detector.detect(text)
        
        print(f"\nReview: {text}")
        print(f"  Critical: {result['has_critical_issue']}")
        print(f"  Severity: {result['severity']}")
        
        if result['issues']:
            for issue in result['issues']:
                print(f"  Issue: {issue['category']} (matched: {issue['matched_keywords']})")
            
            escalation = detector.get_escalation_message(result)
            print(f"  Escalation: {escalation}")


if __name__ == "__main__":
    test_critical_detector()