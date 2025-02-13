document.addEventListener("DOMContentLoaded", function() {
    const manualForm = document.getElementById("predict-form");  // 직접 입력 폼
    const fileForm = document.getElementById("upload-form");  // 파일 업로드 폼
    const manualBtn = document.getElementById("manual-btn");  // 직접 입력 버튼
    const fileBtn = document.getElementById("file-btn");  // 파일 업로드 버튼
    const resultDiv = document.getElementById("result");
    const inputs = document.querySelectorAll("input[type='text'], input[type='number'], input[type='date'], select");
    const fileInput = document.getElementById("file-input");
    const fileNameDisplay = document.getElementById("file-name");  // 선택된 파일명 표시

    console.log("📌 스크립트 로드 완료");

    // 🔹 기본 상태 설정 (직접 입력 폼 활성화)
    manualForm.style.display = "block";
    fileForm.style.display = "none";
    manualBtn.classList.add("button-active");

    // 🔹 "직접 입력" 버튼 클릭 시 (버튼 상태 변경 + 폼 전환)
    manualBtn.addEventListener("click", function() {
        manualForm.style.display = "block";
        fileForm.style.display = "none";
        manualBtn.classList.add("button-active");
        manualBtn.classList.remove("button-inactive");
        fileBtn.classList.add("button-inactive");
        fileBtn.classList.remove("button-active");
    });

    // 🔹 "파일 업로드" 버튼 클릭 시 (버튼 상태 변경 + 폼 전환)
    fileBtn.addEventListener("click", function() {
        manualForm.style.display = "none";
        fileForm.style.display = "block";
        fileBtn.classList.add("button-active");
        fileBtn.classList.remove("button-inactive");
        manualBtn.classList.add("button-inactive");
        manualBtn.classList.remove("button-active");
    });

    // 🔹 파일 선택 시 파일명 표시
    fileInput.addEventListener("change", function () {
        if (fileInput.files.length > 0) {
            fileNameDisplay.textContent = `📂 선택된 파일: ${fileInput.files[0].name}`;
        } else {
            fileNameDisplay.textContent = "선택된 파일 없음";
        }
    });

    // 🔹 직접 입력 폼 제출 처리
    manualForm.addEventListener("submit", function(event) {
        event.preventDefault();

        // ✅ 게재일자 TIMESTAMP 변환 (YYYY-MM-DD → YYYY-MM-DD 00:00:00)
        let dateInput = document.querySelector('input[name="게재일자"]').value;
        let timestampValue = dateInput ? `${dateInput} 00:00:00` : null;

        // ✅ 입력값 가져오기 & JSON 변환
        let formData = {
            "월세": parseFloat(document.querySelector('input[name="월세"]').value),
            "보증금": parseFloat(document.querySelector('input[name="보증금"]').value),
            "관리비": parseFloat(document.querySelector('input[name="관리비"]').value),
            "전용면적": parseFloat(document.querySelector('input[name="전용면적"]').value),
            "방수": parseInt(document.querySelector('input[name="방수"]').value),
            "욕실수": parseInt(document.querySelector('input[name="욕실수"]').value),
            "방향": document.querySelector('select[name="방향"]').value,
            "해당층": parseInt(document.querySelector('input[name="해당층"]').value),
            "총층": parseInt(document.querySelector('input[name="총층"]').value),
            "총주차대수": parseInt(document.querySelector('input[name="총주차대수"]').value),
            "주차가능여부": document.querySelector('select[name="주차가능여부"]').value,
            "제공플랫폼": document.querySelector('input[name="제공플랫폼"]').value,
            "중개사무소": document.querySelector('input[name="중개사무소"]').value,
            "게재일자": timestampValue  // ✅ TIMESTAMP 변환 완료
        };

        // 🔹 서버에 예측 요청 보내기 (POST 요청)
        fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(formData)
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`서버 오류: ${response.status} ${response.statusText}`);
            }
            return response.json();
        })
        .then(data => {
            resultDiv.innerHTML = `<strong>예측 결과:</strong> ${data.prediction}<br><strong>신뢰도:</strong> ${data.confidence}%`;
        })
        .catch(error => {
            console.error("Error:", error);
            resultDiv.innerText = `❌ 예측 중 오류 발생: ${error.message}`;
        });
    });

    // 🔹 파일 업로드 폼 제출 처리
    fileForm.addEventListener("submit", function(event) {
        event.preventDefault();

        if (!fileInput.files.length) {
            alert("파일을 선택해주세요!");
            return;
        }

        let formData = new FormData();
        formData.append("file", fileInput.files[0]);

        fetch("/predict/file", {
            method: "POST",
            body: formData
        })
        .then(response => response.text())
        .then(html => {
            document.open();
            document.write(html);
            document.close();
        })
        .catch(error => {
            console.error("Error:", error);
            resultDiv.innerText = "❌ 파일 예측 중 오류 발생";
        });
    });

    // 🔹 입력 필드 placeholder 동작 & 빨간색 테두리 제거
    inputs.forEach(input => {
        const placeholderText = input.placeholder; // 기존 placeholder 값 저장

        input.addEventListener("focus", function () {
            this.placeholder = "";  // 입력 시 placeholder 제거
            this.classList.remove("input-error"); // 입력하면 빨간색 테두리 제거
        });

        input.addEventListener("blur", function () {
            if (this.value === "") {
                this.placeholder = placeholderText;  // 입력값이 없으면 placeholder 복원
            }
        });
    });
});
