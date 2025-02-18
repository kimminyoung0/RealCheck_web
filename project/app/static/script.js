document.addEventListener("DOMContentLoaded", function () {
    const manualForm = document.getElementById("predict-form");  // 직접 입력 폼
    const fileForm = document.getElementById("upload-form");  // 파일 업로드 폼

    if (!manualForm || !fileForm) {
        console.log("📌 predict.html이 아니라서 실행할 필요 없음.");
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
            body: formData  // ✅ FormData로 전송 (서버에서 JSON을 예상하지 않음)
        })
        .then(response => {
            console.log("📥 서버 응답 상태 코드:", response.status);
            return response.text();  // ✅ HTML을 받아서 처리
        })
        .then(html => {
            document.open();  // ✅ 브라우저의 현재 문서 열기
            document.write(html);  // ✅ 받은 HTML을 브라우저에 렌더링
            document.close();  // ✅ 문서 닫기
        })
        .catch(error => {
            console.error("❌ 예측 중 오류 발생:", error);
            alert("❌ 예측 중 오류가 발생했습니다. 콘솔을 확인하세요.");
        });
    }

    // 🔹 직접 입력 폼 제출 처리 (JSON 대신 FormData 사용)
    manualForm.addEventListener("submit", function (event) {
        event.preventDefault();

        let formData = new FormData();
        formData.append("매물확인방식", document.querySelector('select[name="매물확인방식"]').value);
        formData.append("월세", document.querySelector('input[name="월세"]').value);
        formData.append("보증금", document.querySelector('input[name="보증금"]').value);
        formData.append("관리비", document.querySelector('input[name="관리비"]').value);
        formData.append("전용면적", document.querySelector('input[name="전용면적"]').value);
        formData.append("방수", document.querySelector('input[name="방수"]').value);
        formData.append("욕실수", document.querySelector('input[name="욕실수"]').value);
        formData.append("방향", document.querySelector('select[name="방향"]').value);
        formData.append("해당층", document.querySelector('input[name="해당층"]').value);
        formData.append("총층", document.querySelector('input[name="총층"]').value);
        formData.append("총주차대수", document.querySelector('input[name="총주차대수"]').value);
        formData.append("주차가능여부", document.querySelector('select[name="주차가능여부"]').value);
        formData.append("제공플랫폼", document.querySelector('input[name="제공플랫폼"]').value);
        formData.append("중개사무소", document.querySelector('input[name="중개사무소"]').value);
        formData.append("게재일", document.querySelector('input[name="게재일"]').value + " 00:00:00");

        submitPrediction("/predict", formData); // ✅ fetch 요청 실행
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

        submitPrediction("/predict/file", formData); // ✅ fetch 요청 실행
    });
});
