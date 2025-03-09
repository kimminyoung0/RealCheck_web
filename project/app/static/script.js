document.addEventListener("DOMContentLoaded", function () {
    const manualForm = document.getElementById("predict-form");  // 직접 입력 폼
    const fileForm = document.getElementById("upload-form");  // 파일 업로드 폼

    if (!manualForm || !fileForm) {
        return;
    }

    const fileInput = document.getElementById("file-input");
    const manualBtn = document.getElementById("manual-btn");
    const fileBtn = document.getElementById("file-btn");

    console.log("📌 스크립트 로드 완료");

    // 🔹 기본 상태 설정 (직접 입력 폼 활성화)
    manualForm.style.display = "block";
    fileForm.style.display = "none";
    manualBtn.classList.add("active"); 
    fileBtn.classList.remove("active");

    // 🔹 "직접 입력" 버튼 클릭 시
    manualBtn.addEventListener("click", function () {
        manualForm.style.display = "block";
        fileForm.style.display = "none";
        manualBtn.classList.add("active"); 
        fileBtn.classList.remove("active");
    });

    // 🔹 "파일 업로드" 버튼 클릭 시
    fileBtn.addEventListener("click", function () {
        manualForm.style.display = "none";
        fileForm.style.display = "block";
        fileBtn.classList.add("active");
        manualBtn.classList.remove("active");
    });

    // 🔹 예측 요청을 처리하는 공통 함수 (FormData 방식)
    function submitPrediction(url, formData) {
        fetch(url, {
            method: "POST",
            body: formData  // FormData로 전송 (서버에서 JSON을 예상하지 않음)
        })
        .then(response => {
            console.log("서버 응답 상태 코드:", response.status);
            return response.text();  // HTML을 받아서 처리
        })
        .then(html => {
            document.open();  // 브라우저의 현재 문서 열기
            document.write(html);  // 받은 HTML을 브라우저에 렌더링
            document.close();
        })
        .catch(error => {
            console.error("❌ 예측 중 오류 발생:", error);
            alert("❌ 예측 중 오류가 발생했습니다. 콘솔을 확인하세요.");
        });
    }

    // 🔹 직접 입력 폼 제출 처리
    manualForm.addEventListener("submit", function (event) {
        event.preventDefault(); //기본 동작 제출 방지
        let formData = new FormData(manualForm); // 폼 데이터를 FormData 객체로 저장
        submitPrediction("/input/one", formData); // fetch 요청 실행 (서버에 전송)
    });

    // 🔹 파일 업로드 폼 제출 처리 (FormData 사용)
    fileForm.addEventListener("submit", function (event) {
        event.preventDefault();

        if (!fileInput.files.length) {
            alert("❌ 파일을 선택해주세요!");
            return;
        }

        let formData = new FormData();
        formData.append("file", fileInput.files[0]);

        submitPrediction("/input/file", formData); // fetch 요청 실행
    });
});
