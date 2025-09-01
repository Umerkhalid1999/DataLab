document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('signupForm');
    const errorAlert = document.getElementById('errorAlert');
    const usernameInput = document.getElementById('username');
    const emailInput = document.getElementById('email');
    const passwordInput = document.getElementById('password');

    // Username validation
    function validateUsername() {
        const username = usernameInput.value.trim();
        const isValid = username.length >= 4 && username.length <= 50;
        
        if (isValid) {
            usernameInput.classList.remove('is-invalid');
            usernameInput.classList.add('is-valid');
        } else {
            usernameInput.classList.remove('is-valid');
            usernameInput.classList.add('is-invalid');
        }
        
        return isValid;
    }

    // Enhanced email validation with detailed feedback
    function validateEmail() {
        const email = emailInput.value.trim();
        const errorElement = document.getElementById('emailError');
        const result = window.EmailValidator.validateEmail(email);
        
        // Clear previous states
        emailInput.classList.remove('is-valid', 'is-invalid');
        
        if (email === '') {
            errorElement.textContent = 'Email address is required';
            return false;
        }

        if (result.valid) {
            emailInput.classList.add('is-valid');
            errorElement.textContent = '';
            return true;
        } else {
            emailInput.classList.add('is-invalid');
            errorElement.textContent = result.message;
            
            // Add suggestion if available
            const suggestion = window.EmailValidator.getSuggestion(email);
            if (suggestion) {
                errorElement.innerHTML = `${result.message}<br><small class="text-info">Did you mean: ${email.split('@')[0]}@${suggestion}?</small>`;
            }
            return false;
        }
    }

    // Enhanced password validation
    function validatePassword() {
        const password = passwordInput.value;
        const passwordRegex = /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$/;
        const isValid = passwordRegex.test(password);
        
        if (isValid) {
            passwordInput.classList.remove('is-invalid');
            passwordInput.classList.add('is-valid');
        } else {
            passwordInput.classList.remove('is-valid');
            passwordInput.classList.add('is-invalid');
        }
        
        return isValid;
    }

    // Combined validation
    function validateInputs() {
        const usernameValid = validateUsername();
        const emailValid = validateEmail();
        const passwordValid = validatePassword();
        return usernameValid && emailValid && passwordValid;
    }

    // Add event listeners for real-time validation
    usernameInput.addEventListener('input', validateUsername);
    usernameInput.addEventListener('blur', validateUsername);
    emailInput.addEventListener('input', validateEmail);
    emailInput.addEventListener('blur', validateEmail);
    passwordInput.addEventListener('input', validatePassword);
    passwordInput.addEventListener('blur', validatePassword);

    // Form submission
    form.addEventListener('submit', function(e) {
        e.preventDefault();

        // Clear previous errors
        errorAlert.classList.add('d-none');
        errorAlert.textContent = '';

        // Validate inputs
        if (!validateInputs()) {
            return;
        }

        // Get Firebase auth from the global variable
        const auth = window.firebaseAuth.auth;
        const email = emailInput.value;
        const password = passwordInput.value;
        const username = usernameInput.value;

        window.firebaseAuth.createUser(auth, email, password)
            .then((userCredential) => {
                // Update user profile with username
                return window.firebaseAuth.updateProfile(userCredential.user, {
                    displayName: username
                }).then(() => userCredential);
            })
            .then((userCredential) => {
                // Send email verification
                return window.firebaseAuth.sendEmailVerification(userCredential.user);
            })
            .then(() => {
                // Redirect to login or show verification message
                errorAlert.textContent = 'Account created. Please verify your email.';
                errorAlert.classList.remove('d-none');
                errorAlert.classList.remove('alert-danger');
                errorAlert.classList.add('alert-success');

                // Optional: Redirect after a delay
                setTimeout(() => {
                    window.location.href = "/login";
                }, 2000);
            })
            .catch((error) => {
                // Handle errors
                let errorMessage = 'An error occurred during signup.';
                switch(error.code) {
                    case 'auth/email-already-in-use':
                        errorMessage = 'Email already in use.';
                        break;
                    case 'auth/invalid-email':
                        errorMessage = 'Invalid email address.';
                        break;
                    case 'auth/weak-password':
                        errorMessage = 'Password is too weak.';
                        break;
                    case 'auth/operation-not-allowed':
                        errorMessage = 'Email/password accounts are not enabled.';
                        break;
                }

                errorAlert.textContent = errorMessage;
                errorAlert.classList.remove('d-none');
                errorAlert.classList.add('alert-danger');
            });
    });
});