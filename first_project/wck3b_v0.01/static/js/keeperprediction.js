// 버튼 클릭 시 처리 함수
function makeSelection(selection) {
    // 여기에 선택에 따라 다음 페이지로 이동하는 로직 추가
    // 예: window.location.href = "/next_page?selection=" + selection;
    var modelPrediction = document.getElementById("prediction").textContent;
    // 사용자의 선택과 모델의 예측값 비교 (소문자로 변환하여 비교)
    if (selection.toLowerCase() === modelPrediction.trim().toLowerCase()) {
        // 예측이 맞을 때, 다음 페이지로 이동
        
        window.location.href = "next_page_correct";
    } else {
        // 예측이 틀렸을 때, 다음 페이지로 이동
        window.location.href = "next_page_fail";
    }
}