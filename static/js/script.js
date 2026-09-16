// TourAI Global Script

document.addEventListener("DOMContentLoaded", function () {
    // 1. Tự động đánh dấu active link trên Navbar
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll(".navbar-nav .nav-link");

    navLinks.forEach(link => {
        const href = link.getAttribute("href");
        if (href === currentPath || (href !== "/" && currentPath.startsWith(href))) {
            link.classList.add("active", "fw-bold");
        }
    });

    // 2. Tự động ẩn thông báo Flash sau 5 giây
    const flashAlerts = document.querySelectorAll(".alert-dismissible");
    flashAlerts.forEach(alert => {
        setTimeout(() => {
            const bsAlert = bootstrap.Alert.getOrCreateInstance(alert);
            if (bsAlert) {
                bsAlert.close();
            }
        }, 5000);
    });
});
