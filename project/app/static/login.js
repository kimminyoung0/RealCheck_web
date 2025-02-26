document.addEventListener("DOMContentLoaded", function() {
    const loginForm = document.getElementById("login-form");
    const loginResult = document.getElementById("login-result");

    loginForm.addEventListener("submit", function(event) {
        event.preventDefault();

        let formData = {
            "email": document.querySelector('input[name="email"]').value,
            "password": document.querySelector('input[name="password"]').value
        };

        fetch("/login", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(formData)
        })
        .then(response => response.json())
        .then(data => {
            if (data.token) {
                localStorage.setItem("jwt_token", data.token); // JWT 토큰 저장
                loginResult.innerText = "로그인 성공! 메인 페이지로 이동합니다.";
                
                // ✅ 토큰이 잘 저장되었는지 확인하는 로그 추가
                console.log("🛠️ 저장된 JWT 토큰:", localStorage.getItem("jwt_token"));

                setTimeout(() => {
                    window.location.href = "/"; // 로그인 후 메인 페이지로 이동
                }, 1500);
            } else {
                loginResult.innerText = "로그인 실패: " + data.message;
            }
        })
        .catch(error => {
            console.error("Error:", error);
            loginResult.innerText = "로그인 중 오류가 발생했습니다.";
        });
    });
});
