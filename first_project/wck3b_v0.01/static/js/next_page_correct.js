var video = document.getElementById("fullscreen-video");
// Debugging: Log the video source to the console
console.log("Video Source: ", video.src);

// 비디오가 재생 완료되었을 때 다음 페이지로 이동
video.addEventListener("ended", function() {
    console.log("Video ended. Navigating to the next page.");
    window.location.href = "correct"; // 'final_page'는 최종 페이지의 라우트 이름으로 변경해야 합니다.
});

// 0.01초(10밀리초) 후에 비디오를 재생
setTimeout(function() {
    console.log("Starting video playback after 0.01 seconds.");
    video.play();
}, 10);