"""
Email validation utility for server-side validation
"""
import re
from typing import Dict


class EmailValidator:
    """Comprehensive email validation utility for backend use"""
    
    def __init__(self):
        # More restrictive email regex that prevents invalid domains
        self.email_regex = re.compile(
            r'^[a-zA-Z0-9.!#$%&\'*+/=?^_`{|}~-]+@[a-zA-Z0-9]'
            r'([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?'
            r'(\.[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*\.[a-zA-Z]{2,}$'
        )
        
        # Blocked domains (temporary email services)
        self.blocked_domains = {
            '10minutemail.com', 'tempmail.org', 'guerrillamail.com',
            'mailinator.com', 'throwaway.email', 'temp-mail.org'
        }

    def validate_email(self, email: str) -> Dict[str, any]:
        """Comprehensive email validation"""
        if not email or not email.strip():
            return {
                'valid': False,
                'message': 'Email address is required',
                'code': 'REQUIRED'
            }
        
        email = email.strip().lower()
        
        if not self.email_regex.match(email):
            return {
                'valid': False,
                'message': 'Please enter a valid email format (e.g., user@gmail.com)',
                'code': 'INVALID_FORMAT'
            }
        
        if len(email) > 254:
            return {
                'valid': False,
                'message': 'Email address is too long',
                'code': 'TOO_LONG'
            }
        
        domain_part = email.split('@')[1]
        
        # Check for blocked domains
        if domain_part in self.blocked_domains:
            return {
                'valid': False,
                'message': 'Temporary email addresses are not allowed',
                'code': 'BLOCKED_DOMAIN'
            }
        
        # Check for repeated TLD patterns like .com.com
        domain_parts = domain_part.split('.')
        if len(domain_parts) >= 2:
            common_tlds = ['com', 'org', 'net', 'edu', 'gov', 'mil', 'info', 'biz']
            last_tld = domain_parts[-1]
            second_last_tld = domain_parts[-2]
            
            if last_tld in common_tlds and second_last_tld in common_tlds:
                return {
                    'valid': False,
                    'message': 'Invalid domain format - repeated extensions not allowed',
                    'code': 'REPEATED_TLD'
                }
        
        return {
            'valid': True,
            'message': 'Email address is valid',
            'code': 'VALID'
        }

    def is_valid_email(self, email: str) -> bool:
        """Simple boolean check for email validity"""
        return self.validate_email(email)['valid']


# Global instance for easy import
email_validator = EmailValidator() 