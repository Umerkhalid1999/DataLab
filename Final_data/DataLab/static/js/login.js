document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('loginForm');
    const errorAlert = document.getElementById('errorAlert');
    const emailInput = document.getElementById('email');
    const passwordInput = document.getElementById('password');
    const rememberCheckbox = document.getElementById('remember');

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

    // Password validation
    function validatePassword() {
        const password = passwordInput.value;
        const isValid = password.length >= 8;
        
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
        const emailValid = validateEmail();
        const passwordValid = validatePassword();
        return emailValid && passwordValid;
    }

    // Add event listeners for real-time validation
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

        // Check if Firebase is ready before proceeding
        if (!window.firebaseReady || !window.firebaseAuth) {
            errorAlert.textContent = 'Authentication system is still loading. Please wait a moment and try again.';
            errorAlert.classList.remove('d-none');
            errorAlert.classList.add('alert-warning');
            return;
        }

        // Get Firebase auth from the global variable
        const auth = window.firebaseAuth.auth;
        const email = emailInput.value;
        const password = passwordInput.value;

        // Set persistence based on remember checkbox
        const persistenceType = rememberCheckbox.checked
            ? window.firebaseAuth.persistenceLocal
            : window.firebaseAuth.persistenceSession;

        // Set persistence and sign in
        window.firebaseAuth.setPersistence(auth, persistenceType)
            .then(() => {
                console.log("Persistence set, attempting login...");
                // Attempt login
                return window.firebaseAuth.signIn(auth, email, password);
            })
            .then((userCredential) => {
                console.log("Login successful, checking email verification...");
                // Check email verification
                if (!userCredential.user.emailVerified) {
                    console.log("Email not verified, sending verification email...");
                    // Send verification email again
                    window.firebaseAuth.sendEmailVerification(userCredential.user);

                    errorAlert.textContent = 'Please verify your email. A verification email has been sent.';
                    errorAlert.classList.remove('d-none');
                    errorAlert.classList.remove('alert-danger');
                    errorAlert.classList.add('alert-warning');

                    // Sign out the user
                    return window.firebaseAuth.signOut(auth);
                }

                console.log("User successfully authenticated:", userCredential.user.email);

                // Get ID token
                return userCredential.user.getIdToken().then(idToken => {
                    console.log("ID token retrieved, length:", idToken.length);

                    // Send token to server to create session
                    console.log("Sending token to server...");
                    return fetch('/api/session', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ idToken }),
                        credentials: 'include' // Important for cookies
                    })
                    .then(response => {
                        console.log("Server response status:", response.status);
                        return response.json();
                    })
                    .then(data => {
                        console.log("Session created:", data);
                        if (data.success) {
                            // Redirect to dashboard instead of reloading
                            console.log("Session successful, redirecting to dashboard");
                            window.location.href = '/dashboard';
                        } else {
                            throw new Error(data.message || "Failed to create session");
                        }
                    });
                });
            })
            .catch((error) => {
                console.error("Authentication error:", error);

                // Handle errors
                let errorMessage = 'An error occurred during login.';
                if (error.code) {
                    switch(error.code) {
                        case 'auth/user-not-found':
                            errorMessage = 'No user found with this email.';
                            break;
                        case 'auth/wrong-password':
                            errorMessage = 'Incorrect password.';
                            break;
                        case 'auth/invalid-email':
                            errorMessage = 'Invalid email address.';
                            break;
                        case 'auth/user-disabled':
                            errorMessage = 'This account has been disabled.';
                            break;
                        case 'auth/too-many-requests':
                            errorMessage = 'Too many failed attempts. Please try again later.';
                            break;
                    }
                } else if (error instanceof TypeError && error.message === "Failed to fetch") {
                    console.error("Network error - check server is running and accessible");
                    errorMessage = 'Network error connecting to server. Please try again.';
                } else {
                    errorMessage = error.message || 'Authentication error.';
                }

                errorAlert.textContent = errorMessage;
                errorAlert.classList.remove('d-none');
                errorAlert.classList.add('alert-danger');
            });
    });
});