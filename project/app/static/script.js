document.addEventListener("DOMContentLoaded", function () {
    const manualForm = document.getElementById("predict-form");  // 직접 입력 폼
    const fileForm = document.getElementById("upload-form");  // 파일 업로드 폼
    const resultDiv = document.getElementById("result");
    const fileInput = document.getElementById("file-input");
    const manualBtn = document.getElementById("manual-btn");
    const fileBtn = document.getElementById("file-btn");
    const fileNameDisplay = document.getElementById("file-input");  // 선택된 파일명 표시 (오타 수정)

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

    // 🔹 파일 선택 시 파일명 표시
    fileInput.addEventListener("change", function () {
        if (fileInput.files.length > 0) {
            fileNameDisplay.textContent = `📂 선택된 파일: ${fileInput.files[0].name}`;
        } else {
            fileNameDisplay.textContent = "📁 파일을 선택하세요.";
        }
    });

    // 🔹 직접 입력 폼 제출 처리 (단일 예측)
    manualForm.addEventListener("submit", function (event) {
        event.preventDefault();

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
            "게재일자": document.querySelector('input[name="게재일자"]').value + " 00:00:00"
        };

        fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(formData)
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => {
                    throw new Error(`🚨 서버 오류: ${response.status} - ${err.error}`);
                });
            }
            return response.json();
        })
        .then(data => {
            resultDiv.innerHTML = `<strong>예측 결과:</strong> ${data.prediction}<br><strong>신뢰도:</strong> ${data.pred_proba}%`;
        })
        .catch(error => {
            console.error("❌ 예측 중 오류 발생:", error);
            resultDiv.innerText = `❌ 예측 중 오류 발생: ${error.message}`;
        });
    });

    // 🔹 파일 업로드 폼 제출 처리 (다중 예측)
    fileForm.addEventListener("submit", function (event) {
        event.preventDefault();

        if (!fileInput.files.length) {
            alert("❌ 파일을 선택해주세요!");
            return;
        }

        let formData = new FormData();
        formData.append("file", fileInput.files[0]);


        fetch("/predict/file", {
            method: "POST",
            body: formData
        })
        .then(response => {
            console.log("📥 서버 응답 상태 코드:", response.status);
            return response.text();  // ✅ HTML을 받아서 처리
        })
        .then(html => {
            console.log("📥 서버에서 받은 HTML 응답");
            document.open();  // ✅ 브라우저의 현재 문서 열기
            document.write(html);  // ✅ 받은 HTML을 브라우저에 렌더링
            document.close();  // ✅ 문서 닫기
        })
        .catch(error => {
            console.error("❌ 파일 예측 중 오류 발생:", error);
            alert("❌ 파일 예측 중 오류가 발생했습니다. 콘솔을 확인하세요.");
        });
    });
}); 

// 🔹 예측 결과 표시 함수 (표 형태) → `document.addEventListener` 바깥에 있어야 함!
function displayResults(data) {
    const resultDiv = document.getElementById("result"); // ✅ resultDiv를 함수 내에서 다시 가져옴

    if (!resultDiv) {
        console.error("🚨 `id='result'` 요소를 찾을 수 없습니다. HTML을 확인하세요.");
        return;
    }
    

    if (!data.predictions) {
        resultDiv.innerHTML = `<p style="color:red;">❌ 예측 결과를 받을 수 없습니다.</p>`;
        return;
    }

    let tableHTML = `
        <table border="1" class="table table-striped">
            <thead>
                <tr>
                    <th>번호</th>
                    <th>예측 결과</th>
                    <th>신뢰도 (%)</th>
                </tr>
            </thead>
            <tbody>
    `;

    data.predictions.forEach((pred, index) => {
        tableHTML += `
            <tr>
                <td>${index + 1}</td>
                <td>${pred.prediction}</td>
                <td>${pred.pred_proba}%</td>
            </tr>
        `;
    });

    tableHTML += `</tbody></table>`;

    resultDiv.innerHTML = `<h3>📊 예측 결과</h3> ${tableHTML}`;
}
