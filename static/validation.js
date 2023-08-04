let signup = document.querySelector(".signup");
let login = document.querySelector(".login");
let slider = document.querySelector(".slider");
let formSection = document.querySelector(".form-section");

signup.addEventListener("click", () => {
	slider.classList.add("moveslider");
	formSection.classList.add("form-section-move");
});

login.addEventListener("click", () => {
	slider.classList.remove("moveslider");
	formSection.classList.remove("form-section-move");
});

const form = document.querySelector("form");

form.addEventListener("submit", (event) => {
	event.preventDefault();

	const email = document.querySelector("#email");
	const password = document.querySelector("#password");
	const name = document.querySelector("#name");
	const confirmPassword = document.querySelector("#confirmPassword");

	if (email.checkValidity() && password.checkValidity()) {
		if (form.classList.contains("signup-box")) {
			if (name.checkValidity() && confirmPassword.checkValidity()) {
				console.log("Form submitted successfully!");
			} else {
				console.log("Please enter valid name and confirm password.");
			}
		} else {
			console.log("Form submitted successfully!");
		}
	} else {
		console.log("Please enter valid email and password.");
	}
});
