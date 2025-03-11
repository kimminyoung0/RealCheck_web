document.addEventListener("DOMContentLoaded", function () {
  const historyList = document.getElementById("prediction-history");
  const tableBody = document.querySelector("#prediction-table tbody");

  historyList.addEventListener("click", function (event) {
      if (event.target.tagName === "LI") {
          const predictionId = event.target.dataset.id;
          loadPrediction(predictionId);
      }
  });

  function loadPrediction(id) {
      fetch(`/mypage/prediction/${id}`)
          .then(response => response.json())
          .then(data => {
              document.getElementById("prediction-title").textContent = data.date + " 예측 내역";
              tableBody.innerHTML = "";
              data.records.forEach(record => {
                  const row = `<tr>
                      <td>${record.id}</td>
                      <td>${record.매물확인방식}</td>
                      <td>${record.보증금}</td>
                      <td>${record.전용면적}</td>
                      <td>${record.방수}</td>
                      <td>${record.욕실수}</td>
                      <td>${record.방향}</td>
                      <td>${record.주차가능여부}</td>
                      <td>${record.예측결과}</td>
                      <td>${record.신뢰도}</td>
                  </tr>`;
                  tableBody.innerHTML += row;
              });
          })
          .catch(error => console.error("데이터 로드 오류:", error));
  }
});