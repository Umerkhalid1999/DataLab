// email-validation.js - Comprehensive email validation utility

class EmailValidator {
    constructor() {
        // More restrictive email regex that prevents invalid domains
        this.emailRegex = /^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*\.[a-zA-Z]{2,}$/;
        
        // Common valid email domains
        this.commonDomains = [
            'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'icloud.com',
            'aol.com', 'protonmail.com', 'yandex.com', 'mail.com', 'zoho.com'
        ];
        
        // Blocked domains (temporary email services)
        this.blockedDomains = [
            '10minutemail.com', 'tempmail.org', 'guerrillamail.com', 
            'mailinator.com', 'throwaway.email'
        ];
    }

    /**
     * Comprehensive email validation
     * @param {string} email - Email to validate
     * @returns {object} - Validation result with status and message
     */
    validateEmail(email) {
        // Check if email is provided
        if (!email || email.trim() === '') {
            return {
                valid: false,
                message: 'Email address is required',
                code: 'REQUIRED'
            };
        }

        email = email.trim().toLowerCase();

        // Check basic format
        if (!this.emailRegex.test(email)) {
            return {
                valid: false,
                message: 'Please enter a valid email format (e.g., user@gmail.com)',
                code: 'INVALID_FORMAT'
            };
        }

        // Check email length
        if (email.length > 254) {
            return {
                valid: false,
                message: 'Email address is too long (maximum 254 characters)',
                code: 'TOO_LONG'
            };
        }

        // Split email into local and domain parts
        const [localPart, domainPart] = email.split('@');

        // Validate local part (before @)
        if (localPart.length > 64) {
            return {
                valid: false,
                message: 'Email username part is too long (maximum 64 characters)',
                code: 'LOCAL_TOO_LONG'
            };
        }

        // Check for consecutive dots
        if (localPart.includes('..') || domainPart.includes('..')) {
            return {
                valid: false,
                message: 'Email cannot contain consecutive dots',
                code: 'CONSECUTIVE_DOTS'
            };
        }

        // Check if starts or ends with dot
        if (localPart.startsWith('.') || localPart.endsWith('.')) {
            return {
                valid: false,
                message: 'Email cannot start or end with a dot',
                code: 'INVALID_DOT_POSITION'
            };
        }

        // Validate domain part (after @)
        const domainValidation = this.validateDomain(domainPart);
        if (!domainValidation.valid) {
            return domainValidation;
        }

        // Check for blocked domains
        if (this.blockedDomains.includes(domainPart)) {
            return {
                valid: false,
                message: 'Temporary email addresses are not allowed',
                code: 'BLOCKED_DOMAIN'
            };
        }

        // All validations passed
        return {
            valid: true,
            message: 'Email address is valid',
            code: 'VALID'
        };
    }

    /**
     * Validate domain part of email
     * @param {string} domain - Domain to validate
     * @returns {object} - Validation result
     */
    validateDomain(domain) {
        // Check domain length
        if (domain.length > 253) {
            return {
                valid: false,
                message: 'Email domain is too long',
                code: 'DOMAIN_TOO_LONG'
            };
        }

        // Check if domain has at least one dot
        if (!domain.includes('.')) {
            return {
                valid: false,
                message: 'Email domain must contain at least one dot',
                code: 'INVALID_DOMAIN'
            };
        }

        // Check domain parts
        const domainParts = domain.split('.');
        
        // Check if domain has at least 2 parts
        if (domainParts.length < 2) {
            return {
                valid: false,
                message: 'Email domain format is invalid',
                code: 'INVALID_DOMAIN'
            };
        }

        // Validate each domain part
        for (const part of domainParts) {
            if (part.length === 0) {
                return {
                    valid: false,
                    message: 'Email domain cannot have empty parts',
                    code: 'EMPTY_DOMAIN_PART'
                };
            }
            
            if (part.length > 63) {
                return {
                    valid: false,
                    message: 'Email domain part is too long',
                    code: 'DOMAIN_PART_TOO_LONG'
                };
            }

            // Check domain part characters
            if (!/^[a-zA-Z0-9-]+$/.test(part)) {
                return {
                    valid: false,
                    message: 'Email domain contains invalid characters',
                    code: 'INVALID_DOMAIN_CHARS'
                };
            }

            // Domain parts cannot start or end with hyphen
            if (part.startsWith('-') || part.endsWith('-')) {
                return {
                    valid: false,
                    message: 'Email domain parts cannot start or end with hyphen',
                    code: 'INVALID_DOMAIN_HYPHEN'
                };
            }
        }

        // Check TLD (Top Level Domain) - last part
        const tld = domainParts[domainParts.length - 1];
        if (tld.length < 2) {
            return {
                valid: false,
                message: 'Email domain extension must be at least 2 characters',
                code: 'INVALID_TLD'
            };
        }

        // TLD should only contain letters
        if (!/^[a-zA-Z]+$/.test(tld)) {
            return {
                valid: false,
                message: 'Email domain extension should only contain letters',
                code: 'INVALID_TLD_CHARS'
            };
        }

        // Check for repeated TLD patterns like .com.com, .org.org, etc.
        const commonTlds = ['com', 'org', 'net', 'edu', 'gov', 'mil', 'info', 'biz'];
        if (domainParts.length >= 2) {
            const lastTld = domainParts[domainParts.length - 1];
            const secondLastTld = domainParts[domainParts.length - 2];
            
            // Check if both last parts are common TLDs (indicating invalid pattern like .com.com)
            if (commonTlds.includes(lastTld) && commonTlds.includes(secondLastTld)) {
                return {
                    valid: false,
                    message: 'Invalid domain format - repeated extensions not allowed (e.g., .com.com)',
                    code: 'REPEATED_TLD'
                };
            }
        }

        // Additional check: domains with more than 4 parts are likely invalid
        if (domainParts.length > 4) {
            return {
                valid: false,
                message: 'Domain format is too complex - please use a standard email format',
                code: 'COMPLEX_DOMAIN'
            };
        }

        return {
            valid: true,
            message: 'Domain is valid',
            code: 'VALID'
        };
    }

    /**
     * Check if email domain is from a common provider
     * @param {string} email - Email to check
     * @returns {boolean} - True if from common provider
     */
    isCommonEmailProvider(email) {
        const domain = email.split('@')[1]?.toLowerCase();
        return this.commonDomains.includes(domain);
    }

    /**
     * Get suggestions for typos in email domains
     * @param {string} email - Email to check
     * @returns {string|null} - Suggested correction or null
     */
    getSuggestion(email) {
        const domain = email.split('@')[1]?.toLowerCase();
        const typoMap = {
            'gmai.com': 'gmail.com',
            'gmial.com': 'gmail.com',
            'gmail.co': 'gmail.com',
            'yaho.com': 'yahoo.com',
            'hotmai.com': 'hotmail.com'
        };
        return typoMap[domain] || null;
    }

    /**
     * Real-time validation for input fields
     * @param {HTMLInputElement} inputElement - Email input element
     * @param {HTMLElement} errorElement - Error message element
     * @param {HTMLElement} successElement - Success message element (optional)
     */
    setupRealTimeValidation(inputElement, errorElement, successElement = null) {
        const validator = this;

        function validateAndShowFeedback() {
            const email = inputElement.value.trim();
            const result = validator.validateEmail(email);

            // Clear previous states
            inputElement.classList.remove('is-valid', 'is-invalid');
            if (errorElement) errorElement.textContent = '';
            if (successElement) successElement.textContent = '';

            if (email === '') {
                // Empty field - neutral state
                return;
            }

            if (result.valid) {
                // Valid email
                inputElement.classList.add('is-valid');
                if (successElement) {
                    successElement.textContent = result.message;
                }
            } else {
                // Invalid email
                inputElement.classList.add('is-invalid');
                if (errorElement) {
                    errorElement.textContent = result.message;
                    
                    // Add suggestion if available
                    const suggestion = validator.getSuggestion(email);
                    if (suggestion) {
                        errorElement.innerHTML = `${result.message}<br><small>Did you mean: ${email.split('@')[0]}@${suggestion}?</small>`;
                    }
                }
            }
        }

        // Add event listeners
        inputElement.addEventListener('input', validateAndShowFeedback);
        inputElement.addEventListener('blur', validateAndShowFeedback);
        
        return validateAndShowFeedback;
    }
}

// Create global instance
window.EmailValidator = new EmailValidator(); 