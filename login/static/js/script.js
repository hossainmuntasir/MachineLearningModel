document.addEventListener("DOMContentLoaded", function() {
    const pathName = window.location.pathname;
    const pageName = pathName.split("/").pop();

    console.log("Pathname:", pathName);
    console.log("Page name:", pageName);

    if(pageName === "index") {
        // console.log("Highlighting Home");
        document.querySelector(".home").classList.add("activeLink");
    }
    if(pageName === "about_us") {
        // console.log("Highlighting About Us");
        document.querySelector(".about_us").classList.add("activeLink");
    }
    if(pageName === "building-1") {
        // console.log("Highlighting Building 1");
        document.querySelector(".building_1").classList.add("activeLink");
    }
    if(pageName === "building-2") {
        // console.log("Highlighting Building 2");
        document.querySelector(".building_2").classList.add("activeLink");
    }
    if(pageName === "building-3") {
        // console.log("Highlighting Building 3");
        document.querySelector(".building_3").classList.add("activeLink");
    }
    if(pageName === "model-comparison") {
        // console.log("Highlighting Model Comparison");
        document.querySelector(".model_comparison").classList.add("activeLink");
    }
});
// function fetchPassword() {
//     const username = document.getElementById('username').value;
//     if (username) {
//         fetch(`/get-password?username=${username}`)
//             .then(response => response.json())
//             .then(data => {
//                 document.getElementById('password').value = data.password;
//             });
//     }
// }
function fillPassword() {
    var username = document.getElementById('usernameSelect').value;
    var passwordField = document.getElementById('password');
    var usernameField = document.getElementById('username');  // Input field for the username

    // Hardcoded passwords for each username
    // Change this to  your username and password created
    var passwords = {
        "Building1manager": "EchoEcho",  // user1's password or Building Manager 1
        "Building2manager": "EchoEcho",  // user2's password or Building Manager 2
        "Building3manager": "EchoEcho",  // user3's password or Building Manager 3
        "admin": "EchoEcho"   // admin's password or Administrator
    };

    // If a valid username is selected, auto-fill the password field
    if (passwords[username]) {
        usernameField.value = username; // Autofill the username in the text input
        passwordField.value = passwords[username]; // Autofill the password field
    } else {
        usernameField.value = '';
        passwordField.value = ''; 
    }
}
function validateUsernameSelection() {
    const usernameSelect = document.getElementById('usernameSelect');

    // Show error message if no valid username is selected
    if (!usernameSelect.value) {
        document.querySelector('.msg').innerText = 'Please select a valid username.';
        return false;
    }
    return true; 
}
