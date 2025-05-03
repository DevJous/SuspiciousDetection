(function () {
    // Inserta Font Awesome si no está cargado
    if (!document.querySelector('link[href*="font-awesome"]')) {
        const faLink = document.createElement("link");
        faLink.rel = "stylesheet";
        faLink.href = "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css";
        document.head.appendChild(faLink);
    }

    const css = `
        #toast-container {
            margin-top: 10vh;
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 9999;
        }
        .toast {
            display: flex;
            align-items: center;
            gap: 10px;
            color: #fff;
            padding: 12px 20px;
            margin-bottom: 10px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            opacity: 0;
            transform: translateY(-20px);
            animation: fadeIn 0.4s forwards, fadeOut 0.4s 2.6s forwards;
            font-family: sans-serif;
        }
        .toast.success { background-color: #28a745; }
        .toast.error { background-color: #dc3545; }
        .toast.info { background-color: #007bff; }
        .toast.warning { background-color: #ffc107; color: #000; }
        .toast i {
            font-size: 18px;
        }
        @keyframes fadeIn {
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        @keyframes fadeOut {
            to {
                opacity: 0;
                transform: translateY(-20px);
            }
        }
    `;
    const style = document.createElement("style");
    style.textContent = css;
    document.head.appendChild(style);

    const icons = {
        success: "fa-solid fa-check-circle",
        error: "fa-solid fa-xmark-circle",
        info: "fa-solid fa-circle-info",
        warning: "fa-solid fa-triangle-exclamation"
    };

    window.showToast = function (message, type = "info") {
        let container = document.getElementById("toast-container");
        if (!container) {
            container = document.createElement("div");
            container.id = "toast-container";
            document.body.appendChild(container);
        }

        const toast = document.createElement("div");
        toast.className = `toast ${type}`;

        const icon = document.createElement("i");
        icon.className = icons[type] || icons.info;

        const text = document.createElement("span");
        text.textContent = message;

        toast.appendChild(icon);
        toast.appendChild(text);
        container.appendChild(toast);

        setTimeout(() => {
            toast.remove();
        }, 4000);
    };
})();
