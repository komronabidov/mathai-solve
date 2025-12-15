

// Анимированный фон
const canvas = document.getElementById("bg-canvas");
const ctx = canvas.getContext("2d");

function resize() {
  canvas.width = innerWidth;
  canvas.height = innerHeight;
}

function closeMobileSidebarIfOpen() {
  document.body.classList.remove('mobile-sidebar-open');
  const overlay = document.getElementById('mobile-sidebar-overlay');
  if (overlay) overlay.classList.add('hidden');
}
resize();
window.addEventListener("resize", resize);

const particles = Array.from({ length: innerWidth < 768 ? 40 : 80 }, () => ({
  x: Math.random() * canvas.width,
  y: Math.random() * canvas.height,
  s: Math.random() * 2 + 0.5,
  v: Math.random() * 0.6 + 0.2
}));

(function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "rgba(255,255,255,0.8)";
  particles.forEach(p => {
    ctx.beginPath();
    ctx.arc(p.x, p.y, p.s, 0, Math.PI * 2);
    ctx.fill();
    p.y += p.v;
    if (p.y > canvas.height) p.y = 0;
  });
  requestAnimationFrame(draw);
})();

// Кнопки входа
document.getElementById("loginBtn").addEventListener("click", () => {
  document.getElementById("login-modal").classList.remove("hidden");
});

document.getElementById("signupBtn").addEventListener("click", () => {
  document.getElementById("signup-modal").classList.remove("hidden");
});

// Функция для показа уведомлений
function showNotification(message, type = 'info') {
  const notification = document.createElement('div');
  notification.className = `ai-notification ${type}`;
  notification.textContent = message;
  notification.style.cssText = `
    position: fixed;
    top: 20px;
    right: 20px;
    background: ${type === 'success' ? 'rgba(0, 255, 170, 0.9)' : type === 'error' ? 'rgba(255, 100, 100, 0.9)' : 'rgba(100, 150, 255, 0.9)'};
    color: white;
    padding: 16px 24px;
    border-radius: 12px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    z-index: 10000;
    font-size: 14px;
    font-weight: 500;
    max-width: 400px;
    word-wrap: break-word;
  `;
  document.body.appendChild(notification);

  setTimeout(() => {
    notification.style.opacity = '0';
    notification.style.transition = 'opacity 0.3s ease';
    setTimeout(() => notification.remove(), 300);
  }, 3000);
}

// Закрытие модальных окон
document.querySelectorAll('.modal-close').forEach(btn => {
  btn.addEventListener('click', () => {
    btn.closest('.auth-modal').classList.add('hidden');
  });
});

document.querySelectorAll('.modal-backdrop').forEach(backdrop => {
  backdrop.addEventListener('click', () => {
    backdrop.closest('.auth-modal').classList.add('hidden');
  });
});

// Ultra AI кнопки
function generateRandomProblem() {
  const problems = [
    "Реши уравнение: x² - 5x - 6 = 0",
    "дифференцируй sin(x)",
    "интегрируй x²",
    "площадь круга радиус 5"
  ];
  const input = document.getElementById('main-input');
  if (input) {
    input.value = problems[Math.floor(Math.random() * problems.length)];
    input.focus();
  }
}

function showAISkills() {
  alert(
    "MathAI умеет:\n" +
    "• Решать уравнения и системы\n" +
    "• Решать неравенства (в простых/типовых случаях)\n" +
    "• Дифференцировать и интегрировать\n" +
    "• Упрощать выражения\n" +
    "• Строить графики\n" +
    "• (Опционально) OCR: распознавать задачи по фото\n\n" +
    "Важно (ограничения):\n" +
    "• Нет настоящего LLM-мышления: доказательства, олимпиадные и сложные текстовые задачи решаются только если попали под правила/хэндлеры\n" +
    "• Семантика запроса ограничена: не всегда " +
      "понимает задачу как человек (выделить условия, построить модель)\n" +
    "• Сложные неравенства/системы: SymPy может возвращать сложные множества/условия — иногда требуется дополнительное форматирование и верификация"
  );
}

// Глобальные переменные для чата
const BACKEND_URL = "http://127.0.0.1:5503/api/solve";
let currentChatId = null;
let attachedFile = null;

// Элементы интерфейса
const welcomeScreen = document.getElementById("welcome-screen");
const chatScreen = document.getElementById("chat-screen");
const chatBody = document.getElementById("chat-body");
const historyList = document.getElementById("history-list");
const mobileSidebarOverlay = document.getElementById("mobile-sidebar-overlay");
const mainInput = document.getElementById("main-input");
const mainSendBtn = document.getElementById("main-send");
const chatInput = document.getElementById("chat-input");
const chatSendBtn = document.getElementById("chat-send");
const backBtn = document.getElementById("backBtn");
const newChatBtn = document.getElementById("new-chat-btn");

// Функции переключения экранов
function showWelcome() {
  console.log("Переключаемся на welcome screen");
  closeMobileSidebarIfOpen();
  if (welcomeScreen) {
    welcomeScreen.classList.remove("hidden");
    console.log("Welcome screen показан");
  }
  if (chatScreen) {
    chatScreen.classList.add("hidden");
    console.log("Chat screen скрыт");
  }
}

function showChat() {
  console.log("Переключаемся на chat screen");
  closeMobileSidebarIfOpen();
  if (welcomeScreen) {
    welcomeScreen.classList.add("hidden");
    console.log("Welcome screen скрыт");
  }
  if (chatScreen) {
    chatScreen.classList.remove("hidden");
    console.log("Chat screen показан");
  }
}

// Добавление сообщения в чат
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function appendTextWithLineBreaks(container, text) {
  const parts = String(text ?? '').split(/\r?\n/);
  parts.forEach((part, idx) => {
    container.appendChild(document.createTextNode(part));
    if (idx !== parts.length - 1) container.appendChild(document.createElement('br'));
  });
}

function looksLikeLatex(text) {
  const t = String(text ?? '');
  return /\\[a-zA-Z]+|\^|_|\{|\}|=|\+|\-|\*|\/|\(|\)/.test(t);
}

function normalizeMathForMathJax(text) {
  const t = String(text ?? '').trim();
  if (!t) return '';
  if (/\\\(|\\\)|\\\[|\\\]|\$\$|\$/.test(t)) return t;
  if (!looksLikeLatex(t)) return t;
  return `\\[${t}\\]`;
}

function addMessage(text, isUser = true, imageData = null, plotSrc = null) {
  if (!chatBody) return;

  const wrap = document.createElement("div");
  wrap.className = `msg-wrapper ${isUser ? "user" : "bot"}`;
  const msg = document.createElement("div");
  msg.className = `msg ${isUser ? "user" : "bot"}`;

  // Если есть изображение, добавляем его
  if (imageData) {
    const img = document.createElement("img");
    img.src = imageData;
    img.alt = "Прикрепленное изображение";
    img.style.maxWidth = "100%";
    img.style.borderRadius = "12px";
    img.style.marginBottom = "8px";
    msg.appendChild(img);
  }

  // Добавляем текст (если есть)
  if (text && text.trim()) {
    const textDiv = document.createElement("div");
    const normalized = normalizeMathForMathJax(text);
    appendTextWithLineBreaks(textDiv, normalized);
    msg.appendChild(textDiv);
  }

  if (plotSrc) {
    const plot = document.createElement('img');
    plot.src = plotSrc;
    plot.alt = 'plot';
    plot.style.maxWidth = '100%';
    plot.style.borderRadius = '16px';
    plot.style.margin = '16px 0';
    msg.appendChild(plot);
  }

  wrap.appendChild(msg);
  chatBody.appendChild(wrap);
  chatBody.scrollTop = chatBody.scrollHeight;

  // Обработка MathJax
  if (window.MathJax) {
    MathJax.typesetPromise([msg]).catch((err) => {
      console.log('MathJax error:', err);
    });
  }
}

// Отправка сообщения на сервер
async function sendMessage(text, file = null) {
  if (!text.trim() && !file) return;

  // Переходим в chat screen
  showChat();

  // Если есть файл (изображение), показываем превью в чате
  if (file && file.type.startsWith('image/')) {
    const reader = new FileReader();
    reader.onload = function(e) {
      const imageData = e.target.result;
      // Добавляем сообщение пользователя с изображением
      addMessage(text || "Анализируй изображение", true, imageData);
      // Отправляем на сервер
      sendToServer(text, file);
    };
    reader.readAsDataURL(file);
  } else {
    // Добавляем обычное сообщение пользователя
    addMessage(text, true);
    // Отправляем на сервер
    sendToServer(text, file);
  }

  // Файл очищается после успешной отправки в sendToServer
}

// Отдельная функция для отправки на сервер
async function sendToServer(text, file = null) {
  
  // Сохраняем в Firestore если пользователь авторизован
  if (window.firebaseAuth?.currentUser) {
    await saveMessageToFirestore("user", text);
  }

  // Показываем индикатор загрузки
  const typing = document.createElement("div");
  typing.className = "msg-wrapper bot";
  typing.innerHTML = `<div class="msg bot">•••</div>`;
  chatBody.appendChild(typing);

  try {
    let body, headers;
    if (file) {
      const formData = new FormData();
      formData.append('question', text);
      formData.append('file', file);
      body = formData;
      headers = {};
    } else {
      body = JSON.stringify({ question: text });
      headers = { "Content-Type": "application/json" };
    }

    const res = await fetch(BACKEND_URL, {
      method: "POST",
      headers: headers,
      body: body
    });

    if (res.ok) {
      const data = await res.json();
      typing.remove();

      const latexText = (data && typeof data.latex === 'string') ? data.latex : '';
      const plotSrc = (data && typeof data.plot === 'string') ? data.plot : null;
      addMessage(latexText || (plotSrc ? '' : "Готово!"), false, null, plotSrc);
      
      // Сохраняем ответ в Firestore
      if (window.firebaseAuth?.currentUser) {
        await saveMessageToFirestore("assistant", latexText || "Готово!");
      }
    } else {
      typing.remove();
      addMessage("Ошибка сервера: " + res.status, false);
    }
  } catch (e) {
    console.error("Network error:", e);
    typing.remove();
    addMessage("Сервер недоступен: " + e.message, false);
  }

  // Очищаем файл
  attachedFile = null;
  if (fileInput) fileInput.value = "";
  if (mainFileInput) mainFileInput.value = "";

  // Обновляем превью (скрываем превью контейнер)
  updateFilePreview();
}

// Сохранение сообщения в Firestore
async function saveMessageToFirestore(role, content) {
  if (!window.firebaseAuth || !window.firebaseDb) return;
  const user = window.firebaseAuth.currentUser;
  if (!user) return;
  
  try {
    
    // Создаем чат если его нет
    if (!currentChatId) {
      const chatRef = await window.firestoreAddDoc(
        window.firestoreCollection(window.firebaseDb, "users", user.uid, "chats"),
        {
          title: content.trim().slice(0, 50) || "Новый чат",
          preview: content.trim().slice(0, 60) + (content.length > 60 ? "..." : ""),
          pinned: false,
          createdAt: window.firestoreServerTimestamp(),
          updatedAt: window.firestoreServerTimestamp()
        }
      );
      currentChatId = chatRef.id;
    }
    
    // Сохраняем сообщение
    await window.firestoreAddDoc(
      window.firestoreCollection(window.firebaseDb, "users", user.uid, "chats", currentChatId, "messages"),
      {
        role: role,
        content: content,
        timestamp: window.firestoreServerTimestamp()
      }
    );
    
    // Обновляем чат
    await window.firestoreUpdateDoc(
      window.firestoreDoc(window.firebaseDb, "users", user.uid, "chats", currentChatId),
      {
        title: role === "user" ? (content.trim().slice(0, 50) || "Новый чат") : undefined,
        preview: role === "user" ? (content.trim().slice(0, 60) + (content.length > 60 ? "..." : "")) : undefined,
        updatedAt: window.firestoreServerTimestamp()
      }
    );
  } catch (error) {
    console.error("Ошибка сохранения в Firestore:", error);
  }
}

// Загрузка чата из Firestore
async function loadChat(chatId) {
  // Если пользователь авторизован, загружаем чат из Firestore
  if (window.firebaseAuth && window.firebaseDb && window.firebaseAuth.currentUser) {
    if (currentChatId === chatId) return;
    const user = window.firebaseAuth.currentUser;

    currentChatId = chatId;
    if (chatBody) chatBody.innerHTML = "<div class='msg-wrapper bot'><div class='msg bot'>Загрузка...</div></div>";

    try {
      const q = window.firestoreQuery(
        window.firestoreCollection(window.firebaseDb, "users", user.uid, "chats", chatId, "messages"),
        window.firestoreOrderBy("timestamp", "asc")
      );
      const snap = await window.firestoreGetDocs(q);

      if (chatBody) chatBody.innerHTML = "";
      snap.forEach(d => {
        const data = d.data();
        addMessage(data.content, data.role === "user");
      });

      showChat();
      if (chatBody) chatBody.scrollTop = chatBody.scrollHeight;
    } catch (error) {
      console.error("Ошибка загрузки чата:", error);
      if (chatBody) chatBody.innerHTML = "<div class='msg-wrapper bot'><div class='msg bot'>Ошибка загрузки</div></div>";
    }
  } else {
    // Если пользователь не авторизован, просто переходим в пустой чат
    currentChatId = null;
    if (chatBody) chatBody.innerHTML = "";
    showChat();
  }
}

// Создание нового чата
async function createNewChat() {
  console.log("Создание нового чата");
  currentChatId = null;
  if (chatBody) {
    chatBody.innerHTML = "";
    console.log("Chat body очищен");
  }
  showChat();
}

// Рендеринг истории чатов
function renderHistory(snap) {
  if (!historyList) return;
  
  historyList.innerHTML = snap.empty ? "<div class='no-chats'>Нет чатов</div>" : "";
  
  const pinned = [], normal = [];
  snap.forEach(d => {
    const data = d.data();
    (data.pinned ? pinned : normal).push({ id: d.id, ...data });
  });
  
  [...pinned, ...normal].forEach(c => {
    const el = document.createElement("div");
    el.className = "history-item";
    el.dataset.id = c.id;
    if (c.id === currentChatId) el.classList.add("active");

    const safeTitle = escapeHtml(c.title || "Новый чат");
    const safePreview = escapeHtml(c.preview || "");
    
    el.innerHTML = `
      <div class="history-item-content">
        <div class="chat-info">
          <div class="chat-title">${safeTitle}</div>
          <div class="chat-date">${safePreview}</div>
        </div>
      </div>
      <button class="menu-dots" aria-label="Меню"></button>
      <div class="chat-menu">
        <button class="menu-item rename">Переименовать</button>
        <button class="menu-item pin">${c.pinned ? "Открепить" : "Закрепить"}</button>
        <button class="menu-item delete">Удалить</button>
      </div>
    `;
    
    el.querySelector(".history-item-content").onclick = () => loadChat(c.id);
    
    // Меню с тремя точками
    const dots = el.querySelector(".menu-dots");
    const menu = el.querySelector(".chat-menu");
    
    dots.onclick = (e) => {
      e.stopPropagation();
      const isOpen = menu.classList.contains("show");
      closeAllMenus(e);
      if (!isOpen) {
        setTimeout(() => menu.classList.add("show"), 10);
      }
    };
    
    // Переименование
    el.querySelector(".rename").onclick = async (e) => {
      e.stopPropagation();
      closeAllMenus(e);
      const titleEl = el.querySelector(".chat-title");
      const oldTitle = titleEl.textContent;
      const input = document.createElement("input");
      input.type = "text";
      input.value = oldTitle;
      input.style.cssText = `
        width: 100%;
        background: rgba(255,255,255,0.1);
        border: 1px solid rgba(255,255,255,0.3);
        border-radius: 8px;
        padding: 6px 10px;
        color: white;
        font-size: 14.5px;
        font-weight: 500;
        outline: none;
        backdrop-filter: blur(10px);
      `;
      titleEl.textContent = "";
      titleEl.appendChild(input);
      input.focus();
      input.select();
      
      const save = async () => {
        const newTitle = input.value.trim();
        if (newTitle && newTitle !== oldTitle && window.firebaseAuth?.currentUser) {
          try {
            await window.firestoreUpdateDoc(
              window.firestoreDoc(window.firebaseDb, "users", window.firebaseAuth.currentUser.uid, "chats", c.id),
              { title: newTitle }
            );
          } catch (error) {
            console.error("Ошибка переименования:", error);
          }
        }
        titleEl.textContent = newTitle || oldTitle;
      };
      
      input.addEventListener("blur", save);
      input.addEventListener("keydown", (ev) => {
        if (ev.key === "Enter") { ev.preventDefault(); input.blur(); }
        if (ev.key === "Escape") { titleEl.textContent = oldTitle; }
      });
    };
    
    // Закрепление
    el.querySelector(".pin").onclick = async (e) => {
      e.stopPropagation();
      closeAllMenus(e);
      if (window.firebaseAuth?.currentUser) {
        try {
          await window.firestoreUpdateDoc(
            window.firestoreDoc(window.firebaseDb, "users", window.firebaseAuth.currentUser.uid, "chats", c.id),
            { pinned: !c.pinned }
          );
        } catch (error) {
          console.error("Ошибка закрепления:", error);
        }
      }
    };
    
    // Удаление
    el.querySelector(".delete").onclick = (e) => {
      e.stopPropagation();
      closeAllMenus(e);
      showDeleteModal(c.id, c.title || "Новый чат");
    };
    
    historyList.appendChild(el);
  });
}

// Закрытие всех меню
function closeAllMenus(e) {
  // Не закрываем если клик внутри меню
  if (e && (e.target.closest('.chat-menu') || e.target.closest('.menu-dots'))) {
    return;
  }
  document.querySelectorAll(".chat-menu.show").forEach(m => m.classList.remove("show"));
}
document.addEventListener("click", closeAllMenus);

// Модалка удаления
let chatToDelete = null;

// Функция экспорта чата
function exportChat() {
  const chatBody = document.getElementById('chat-body');
  if (!chatBody || !chatBody.innerText.trim()) {
    showNotification("Чат пустой", "warning");
    return;
  }

  const messages = chatBody.querySelectorAll('.msg-wrapper');
  let chatText = "Math AI Chat Export\n" + new Date().toLocaleString() + "\n\n";

  messages.forEach(wrapper => {
    const msg = wrapper.querySelector('.msg');
    if (msg) {
      const isUser = wrapper.classList.contains('user');
      const sender = isUser ? "Вы: " : "AI: ";
      chatText += sender + msg.innerText + "\n\n";
    }
  });

  const blob = new Blob([chatText], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `math-ai-chat-${new Date().toISOString().split('T')[0]}.txt`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);

  showNotification("Чат экспортирован", "success");
}
function showDeleteModal(chatId, chatTitle) {
  chatToDelete = chatId;
  const modal = document.getElementById("delete-modal");
  if (modal) {
    modal.classList.add("show");
    const titleEl = modal.querySelector(".delete-modal-title");
    if (titleEl) titleEl.textContent = `Удалить "${chatTitle}"?`;
  }
}

async function confirmDelete() {
  if (chatToDelete && window.firebaseAuth?.currentUser) {
    try {
      await window.firestoreDeleteDoc(
        window.firestoreDoc(window.firebaseDb, "users", window.firebaseAuth.currentUser.uid, "chats", chatToDelete)
      );
      if (currentChatId === chatToDelete) {
        currentChatId = null;
        if (chatBody) chatBody.innerHTML = "";
        showWelcome();
      }
      chatToDelete = null;
      const modal = document.getElementById("delete-modal");
      if (modal) modal.classList.remove("show");
    } catch (error) {
      console.error("Ошибка удаления:", error);
    }
  }
}

// Подписка на изменения чатов
function listenToChats() {
  if (!window.firebaseAuth || !window.firebaseDb) return;
  const user = window.firebaseAuth.currentUser;
  if (!user) return;
  const q = window.firestoreQuery(
    window.firestoreCollection(window.firebaseDb, "users", user.uid, "chats"),
    window.firestoreOrderBy("updatedAt", "desc")
  );
  
  window.firestoreOnSnapshot(q, renderHistory);
}

// Кнопка выхода (добавляется после загрузки DOM)

// Инициализация
document.addEventListener('DOMContentLoaded', () => {
  // Обработчики Ultra AI кнопок
  document.querySelectorAll('.ultra-btn').forEach(btn => {
    const action = btn.getAttribute('data-action');
    if (action === 'random') {
      btn.addEventListener('click', generateRandomProblem);
    } else if (action === 'skills') {
      btn.addEventListener('click', showAISkills);
    }
  });

  // Обработчики подсказок
  document.querySelectorAll('.example-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      const exampleText = btn.dataset.example;
      const input = document.getElementById('main-input');
      if (input && exampleText) {
        input.value = exampleText;
        input.focus();
      }
    });
  });

  // Google кнопки - добавляем обработчики после загрузки DOM
  document.querySelectorAll("#googleLogin, #googleSignup").forEach(btn => {
    btn.addEventListener("click", async () => {
      // Ждем инициализации Firebase (может потребоваться небольшая задержка)
      let attempts = 0;
      while (!window.firebaseAuth || !window.firebaseProvider) {
        if (attempts++ > 10) {
          alert("Firebase не загружен. Обновите страницу.");
          return;
        }
        await new Promise(resolve => setTimeout(resolve, 100));
      }

      try {
        const result = await window.signInWithPopup(window.firebaseAuth, window.firebaseProvider);
        console.log("✅ Google вход успешен:", result.user.email);
        
        // Закрываем модальное окно
        btn.closest('.auth-modal').classList.add('hidden');
        
        // Показываем успешное сообщение
        const userName = result.user.displayName || result.user.email;
        showNotification("Вход выполнен успешно! Добро пожаловать, " + userName, "success");
        
      } catch (error) {
        console.error("❌ Ошибка Google входа:", error);
        let errorMessage = "Неизвестная ошибка";

        if (error.code === 'auth/popup-blocked') {
          errorMessage = "Всплывающее окно заблокировано браузером. Разрешите всплывающие окна для этого сайта.";
        } else if (error.code === 'auth/popup-closed-by-user') {
          errorMessage = "Окно входа было закрыто";
        } else if (error.code === 'auth/cancelled-popup-request') {
          errorMessage = "Запрос на вход был отменен";
        } else if (error.code === 'auth/network-request-failed') {
          errorMessage = "Проблема с сетью. Проверьте подключение к интернету.";
        } else {
          errorMessage = error.message || "Ошибка входа";
        }

        showNotification("Ошибка входа: " + errorMessage, "error");
      }
    });
  });

  // Обработчики отправки сообщений
  if (mainSendBtn && mainInput) {
    mainSendBtn.addEventListener('click', () => {
      const text = mainInput.value.trim();
      if (text || attachedFile) {
        sendMessage(text || "Анализируй прикреплённый файл", attachedFile);
        mainInput.value = "";
      }
    });
    
    mainInput.addEventListener('keydown', (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        const text = mainInput.value.trim();
        if (text || attachedFile) {
          sendMessage(text || "Анализируй прикреплённый файл", attachedFile);
          mainInput.value = "";
        }
      }
    });
  }

  if (chatSendBtn && chatInput) {
    chatSendBtn.addEventListener('click', () => {
      const text = chatInput.value.trim();
      if (text || attachedFile) {
        sendMessage(text || "Анализируй прикреплённый файл", attachedFile);
        chatInput.value = "";
      }
    });
    
    chatInput.addEventListener('keydown', (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        const text = chatInput.value.trim();
        if (text || attachedFile) {
          sendMessage(text || "Анализируй прикреплённый файл", attachedFile);
          chatInput.value = "";
        }
      }
    });
  }

  // Кнопка назад
  if (backBtn) {
    backBtn.addEventListener('click', showWelcome);
  }

  // Кнопка нового чата
  if (newChatBtn) {
    newChatBtn.addEventListener('click', (e) => {
      console.log("Клик на кнопку нового чата");
      e.stopPropagation();
      createNewChat();
    });
  }

  // Обработчик клика на весь sidebar для создания нового чата
  const sidebar = document.getElementById("sidebar");
  if (sidebar) {
    sidebar.addEventListener('click', (e) => {
      // Если клик был не на элементе истории чата и не на меню, создаем новый чат
      if (!e.target.closest('.history-item') &&
          !e.target.closest('.chat-menu') &&
          !e.target.closest('.menu-dots') &&
          !e.target.closest('#new-chat-btn') &&
          !e.target.closest('#sidebar-toggle-btn') &&
          !e.target.closest('#user-avatar-btn') &&
          !e.target.closest('.search-container') &&
          e.target.id !== 'sidebar') {
        console.log("Клик на sidebar - создаем новый чат");
        createNewChat();
      }
    });
  }

  // Кнопка переключения sidebar
  const sidebarToggleBtn = document.getElementById("sidebar-toggle-btn");
  const collapsedNewChat = document.getElementById("collapsed-new-chat");
  const collapsedSearch = document.getElementById("collapsed-search");
  const collapsedToggleBtn = document.getElementById("collapsed-toggle-btn");

  function isMobile() {
    return window.matchMedia && window.matchMedia('(max-width: 768px)').matches;
  }

  function openMobileSidebar() {
    document.body.classList.add('mobile-sidebar-open');
    if (mobileSidebarOverlay) mobileSidebarOverlay.classList.remove('hidden');
  }

  function closeMobileSidebar() {
    closeMobileSidebarIfOpen();
  }

  function toggleMobileSidebar() {
    if (document.body.classList.contains('mobile-sidebar-open')) closeMobileSidebar();
    else openMobileSidebar();
  }

  if (mobileSidebarOverlay) {
    mobileSidebarOverlay.addEventListener('click', () => {
      closeMobileSidebarIfOpen();
    });
  }

  window.addEventListener('resize', () => {
    if (!isMobile()) {
      closeMobileSidebarIfOpen();
    }
  });
  
  function updateSidebarToggleIcon() {
    if (sidebarToggleBtn) {
      const isCollapsed = document.body.classList.contains("sidebar-collapsed");
      const closeIcon = sidebarToggleBtn.querySelector(".toggle-icon-close");
      const openIcon = sidebarToggleBtn.querySelector(".toggle-icon-open");
      if (closeIcon && openIcon) {
        closeIcon.style.display = isCollapsed ? "none" : "block";
        openIcon.style.display = isCollapsed ? "block" : "none";
      }
    }
  }

  // Функция для обновления позиции кнопки Back в зависимости от состояния сайдбара
  function updateBackButtonPosition() {
    // CSS автоматически обработает позиционирование через классы body
    // Никаких дополнительных действий не требуется
  }
  
  if (sidebarToggleBtn) {
    sidebarToggleBtn.addEventListener('click', () => {
      if (isMobile()) {
        toggleMobileSidebar();
      } else {
        document.body.classList.toggle("sidebar-collapsed");
        updateSidebarToggleIcon();
        updateBackButtonPosition();
      }
    });
  }
  
  
  if (collapsedNewChat) {
    collapsedNewChat.addEventListener('click', (e) => {
      console.log("Клик на collapsed кнопку нового чата");
      e.stopPropagation();
      createNewChat();
    });
  }
  
  if (collapsedSearch) {
    collapsedSearch.addEventListener('click', () => {
      if (isMobile()) {
        openMobileSidebar();
      } else {
        document.body.classList.remove("sidebar-collapsed");
        updateSidebarToggleIcon();
        updateBackButtonPosition();
      }
      const chatSearch = document.getElementById("chat-search");
      if (chatSearch) setTimeout(() => chatSearch.focus(), 300);
    });
  }
  
  
  if (collapsedToggleBtn) {
    collapsedToggleBtn.addEventListener('click', () => {
      if (isMobile()) {
        openMobileSidebar();
      } else {
        document.body.classList.remove("sidebar-collapsed");
        updateSidebarToggleIcon();
        updateBackButtonPosition();
      }
    });
  }
  
  // Отслеживание изменений состояния sidebar
  const observer = new MutationObserver(() => {
    updateSidebarToggleIcon();
    updateBackButtonPosition();
  });
  observer.observe(document.body, { attributes: true, attributeFilter: ["class"] });
  updateSidebarToggleIcon();
  updateBackButtonPosition();

  // Модалка удаления
  const deleteModal = document.getElementById("delete-modal");
  const cancelDeleteBtn = document.getElementById("cancel-delete");
  const confirmDeleteBtn = document.getElementById("confirm-delete");
  
  if (cancelDeleteBtn) {
    cancelDeleteBtn.addEventListener('click', () => {
      chatToDelete = null;
      if (deleteModal) deleteModal.classList.remove("show");
    });
  }
  
  if (confirmDeleteBtn) {
    confirmDeleteBtn.addEventListener('click', confirmDelete);
  }
  
  if (deleteModal) {
    const backdrop = deleteModal.querySelector(".delete-modal-backdrop");
    if (backdrop) {
      backdrop.addEventListener('click', () => {
        chatToDelete = null;
        deleteModal.classList.remove("show");
      });
    }
  }

  // Функция для обновления превью файла
function updateFilePreview() {
  const mainPreview = document.getElementById("main-file-preview");
  const chatPreview = document.getElementById("chat-file-preview");

  if (attachedFile) {
    const fileName = attachedFile.name;
    const fileSize = (attachedFile.size / 1024).toFixed(2) + " MB";

    // Обновляем превью для welcome screen
    if (mainPreview) {
      mainPreview.style.display = "block";
      const img = document.getElementById("main-file-preview-img");
      const name = document.getElementById("main-file-preview-name");
      const size = document.getElementById("main-file-preview-size");

      if (attachedFile.type.startsWith('image/')) {
        // Для изображений показываем превью
        const reader = new FileReader();
        reader.onload = function(e) {
          if (img) img.src = e.target.result;
        };
        reader.readAsDataURL(attachedFile);
        if (img) img.style.display = "block";
      } else {
        // Для других файлов скрываем изображение
        if (img) img.style.display = "none";
      }

      if (name) name.textContent = fileName;
      if (size) size.textContent = fileSize;
    }

    // Обновляем превью для chat screen
    if (chatPreview) {
      chatPreview.style.display = "block";
      const img = document.getElementById("chat-file-preview-img");
      const name = document.getElementById("chat-file-preview-name");
      const size = document.getElementById("chat-file-preview-size");

      if (attachedFile.type.startsWith('image/')) {
        // Для изображений показываем превью
        const reader = new FileReader();
        reader.onload = function(e) {
          if (img) img.src = e.target.result;
        };
        reader.readAsDataURL(attachedFile);
        if (img) img.style.display = "block";
      } else {
        // Для других файлов скрываем изображение
        if (img) img.style.display = "none";
      }

      if (name) name.textContent = fileName;
      if (size) size.textContent = fileSize;
    }
  } else {
    // Скрываем превью
    if (mainPreview) mainPreview.style.display = "none";
    if (chatPreview) chatPreview.style.display = "none";
  }
}

  // Обработка файлов
  const fileInput = document.getElementById("file-input");
  const mainFileInput = document.getElementById("main-file-input");
  const fileUploadBtn = document.getElementById("file-upload-btn");
  const mainFileUploadBtn = document.getElementById("main-file-upload-btn");
  const mainPreviewRemove = document.getElementById("main-file-preview-remove");
  const chatPreviewRemove = document.getElementById("chat-file-preview-remove");

  if (fileUploadBtn && fileInput) {
    fileUploadBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (e) => {
      attachedFile = e.target.files[0];
      updateFilePreview();
    });
  }

  if (mainFileUploadBtn && mainFileInput) {
    mainFileUploadBtn.addEventListener('click', () => mainFileInput.click());
    mainFileInput.addEventListener('change', (e) => {
      attachedFile = e.target.files[0];
      updateFilePreview();
    });
  }

  // Обработчики кнопок удаления файла в превью
  if (mainPreviewRemove) {
    mainPreviewRemove.addEventListener('click', () => {
      attachedFile = null;
      updateFilePreview();
      // Очищаем input file
      if (mainFileInput) mainFileInput.value = '';
    });
  }

  if (chatPreviewRemove) {
    chatPreviewRemove.addEventListener('click', () => {
      attachedFile = null;
      updateFilePreview();
      // Очищаем input file
      if (fileInput) fileInput.value = '';
    });
  }

  // Обработка состояния аутентификации (после загрузки Firebase)
  setTimeout(() => {
    if (window.onAuthStateChanged && window.firebaseAuth) {
      window.onAuthStateChanged(window.firebaseAuth, (user) => {
        console.log("Auth state changed:", user ? user.email : "signed out");
        
        const sidebar = document.getElementById("sidebar");
        const topAuthButtons = document.getElementById("top-auth-buttons");
        const userAvatar = document.getElementById("user-avatar");
        const userAvatarBtn = document.getElementById("user-avatar-btn");
        
        if (user) {
          // Пользователь вошел
          if (sidebar) sidebar.classList.remove("hidden");
          if (topAuthButtons) topAuthButtons.classList.add("hidden");
          if (userAvatar && user.photoURL) userAvatar.src = user.photoURL;
          if (userAvatarBtn) userAvatarBtn.style.display = "flex";
          document.body.classList.remove("sidebar-collapsed");
          updateBackButtonPosition();
          
          // Загружаем историю чатов
          listenToChats();
        } else {
          // Пользователь вышел
          if (sidebar) sidebar.classList.add("hidden");
          if (topAuthButtons) topAuthButtons.classList.remove("hidden");
          if (userAvatar) userAvatar.src = "";
          if (userAvatarBtn) userAvatarBtn.style.display = "none";
          document.body.classList.add("sidebar-collapsed");
          updateBackButtonPosition();
          currentChatId = null;
          if (historyList) historyList.innerHTML = "<div class='no-chats'>Войдите</div>";
        }
      });
    }

    // Кнопка выхода
    const logoutBtn = document.querySelector('.logout');
    if (logoutBtn && window.signOut && window.firebaseAuth) {
      logoutBtn.addEventListener('click', () => {
        window.signOut(window.firebaseAuth);
        showNotification("Вы вышли из аккаунта", "info");
        currentChatId = null;
        if (chatBody) chatBody.innerHTML = "";
        showWelcome();
      });
    }

    // Функция показа/скрытия примеров в чате
    function toggleChatExamples() {
      const chatBody = document.getElementById('chat-body');
      const examplesSection = document.getElementById('chat-examples-section');

      if (chatBody && examplesSection) {
        const hasMessages = chatBody.querySelector('.msg-wrapper');
        examplesSection.style.display = hasMessages ? 'none' : 'block';
      }
    }

    // Наблюдатель за изменениями в чате
    const chatObserver = new MutationObserver(toggleChatExamples);
    const chatBody = document.getElementById('chat-body');
    if (chatBody) {
      chatObserver.observe(chatBody, {
        childList: true,
        subtree: true
      });
    }

    // Первоначальная проверка
    toggleChatExamples();

    // Обработчики для примеров в чате
    document.addEventListener('click', (e) => {
      if (e.target.closest('#chat-examples-section .example-btn')) {
        const btn = e.target.closest('.example-btn');
        const example = btn.getAttribute('data-example');
        const input = document.getElementById('chat-input');
        if (input && example) {
          input.value = example;
          input.focus();
          // Автоматически отправляем сообщение
          setTimeout(() => {
            const sendBtn = document.getElementById('chat-send');
            if (sendBtn) sendBtn.click();
          }, 100);
        }
      }

      if (e.target.closest('.ultra-btn[data-action="clear"]')) {
        if (confirm('Очистить весь чат? Это действие нельзя отменить.')) {
          if (chatBody) chatBody.innerHTML = "";
          toggleChatExamples(); // Обновляем видимость примеров
          showNotification("Чат очищен", "success");
        }
      }

      if (e.target.closest('.ultra-btn[data-action="export"]')) {
        exportChat();
      }
    });
  }, 500);

  // Инициализация превью файлов
  updateFilePreview();

  



let currentAttachedFile = null;

function showAttachedPreview(file) {
  // Находим правильный контейнер (работает и на welcome, и в чате)
  const inputEl = document.getElementById('chat-input') || document.getElementById('main-input');
  if (!inputEl) return;
  
  const wrapper = inputEl.closest('.grok-input') || inputEl.closest('.chat-input-container') || inputEl.parentElement;
  if (!wrapper) return;

  // Удаляем все старые превью
  wrapper.querySelectorAll('.attached-preview, #main-file-preview, #chat-file-preview').forEach(el => el.remove());

  const preview = document.createElement('div');
  preview.className = 'attached-preview';
  preview.style.cssText = `
    position: absolute;
    left: 12px; right: 12px; bottom: 100%;
    margin-bottom: 8px;
    padding: 10px 14px;
    background: rgba(40,40,50,0.95);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 14px;
    display: flex;
    align-items: center;
    gap: 10px;
    z-index: 10;
    backdrop-filter: blur(16px);
    box-shadow: 0 -6px 20px rgba(0,0,0,0.4);
  `;

  const img = document.createElement('img');
  img.style.cssText = 'width: 44px; height: 44px; object-fit: cover; border-radius: 8px; flex-shrink: 0;';
  if (file.type.startsWith('image/')) {
    const reader = new FileReader();
    reader.onload = e => img.src = e.target.result;
    reader.readAsDataURL(file);
  } else {
    img.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDQiIGhlaWdodD0iNDQiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PHJlY3Qgd2lkdGg9IjQ0IiBoZWlnaHQ9IjQ0IiBmaWxsPSIjNDA0MDQwIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtc2l6ZT0iMTgiIGZpbGw9IiNmZmYiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGR5PSIuM2VtIj7wn5mDPC90ZXh0Pjwvc3ZnPg==';
  }

  const name = document.createElement('span');
  name.textContent = file.name.length > 28 ? file.name.slice(0,25)+'...' : file.name;
  name.style.cssText = 'color: white; font-size: 14px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;';

  const removeBtn = document.createElement('button');
  removeBtn.innerHTML = '×';
  removeBtn.style.cssText = `
    margin-left: auto; background: rgba(255,80,80,0.3); border: none; color: white;
    width: 26px; height: 26px; border-radius: 50%; cursor: pointer;
    font-size: 18px; font-weight: bold; display: flex; align-items: center; justify-content: center;
  `;
  removeBtn.onclick = e => {
    e.stopPropagation();
    currentAttachedFile = null;
    preview.remove();
    document.getElementById('main-file-input').value = '';
    document.getElementById('file-input').value = '';
  };

  preview.append(img, name, removeBtn);
  wrapper.style.position = 'relative';
  wrapper.insertBefore(preview, wrapper.firstChild);
}

// Скрепки
document.querySelectorAll('#main-file-upload-btn, #file-upload-btn').forEach(btn => {
  btn.onclick = e => {
    e.stopPropagation();
    const id = btn.id.includes('main') ? 'main-file-input' : 'file-input';
    document.getElementById(id)?.click();
  };
});

// Файловые input'ы
document.querySelectorAll('#main-file-input, #file-input').forEach(input => {
  input.addEventListener('change', e => {
    if (e.target.files[0]) {
      currentAttachedFile = e.target.files[0];
      showAttachedPreview(currentAttachedFile);
    }
  });
});

// Drag & Drop
document.querySelectorAll('.grok-input, .chat-input, .chat-input-container').forEach(container => {
  ['dragenter', 'dragover'].forEach(ev => {
    container.addEventListener(ev, e => e.preventDefault());
  });
  container.addEventListener('drop', e => {
    e.preventDefault();
    if (e.dataTransfer.files[0]) {
      currentAttachedFile = e.dataTransfer.files[0];
      const inputId = document.getElementById('chat-input') ? 'file-input' : 'main-file-input';
      const fileInput = document.getElementById(inputId);
      if (fileInput) {
        const dt = new DataTransfer();
        dt.items.add(currentAttachedFile);
        fileInput.files = dt.files;
      }
      showAttachedPreview(currentAttachedFile);
    }
  });
});

// Перехват отправки — используем актуальный файл
const oldSendMessage = window.sendMessage;
window.sendMessage = function(text = '', file = null) {
  const fileToSend = file || currentAttachedFile || null;
  oldSendMessage(text, fileToSend);

  // Убираем превью после отправки
  setTimeout(() => {
    currentAttachedFile = null;
    document.querySelectorAll('.attached-preview, #main-file-preview, #chat-file-preview').forEach(el => el.remove());
  }, 300);
};

// Перехват кнопок отправки и Enter
document.addEventListener('click', e => {
  if (e.target.matches('#main-send, #chat-send, #main-send *, #chat-send *')) {
    const textarea = document.getElementById('main-input') || document.getElementById('chat-input');
    if (textarea) {
      const text = textarea.value.trim();
      window.sendMessage(text);
      textarea.value = '';
    }
  }
});

document.addEventListener('keydown', e => {
  if ((e.target.id === 'main-input' || e.target.id === 'chat-input') && e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    window.sendMessage(e.target.value.trim());
    e.target.value = '';
  }
});


// === ДОБАВЬ ЭТОТ КОД В КОНЕЦ ФАЙЛА ===

// Импорты (если их ещё нет в файле — добавь наверху файла)
import { getRedirectResult } from "firebase/auth";

// Этот useEffect обработает результат после возврата с Google
useEffect(() => {
  if (typeof window !== "undefined") { // важно для Next.js
    getRedirectResult(auth)
      .then((result) => {
        if (result?.user) {
          console.log("Залогинен через redirect:", result.user);
          // Здесь можно обновить состояние, редиректнуть и т.д.
          // Например: router.push("/dashboard") если используешь Next.js
        }
      })
      .catch((error) => {
        console.error("Ошибка redirect:", error);
      });
  }
}, []);

// Переопределяем поведение кнопки Google: вместо popup используем redirect
// Найди свою функцию или кнопку, которая вызывает signInWithPopup, и замени её на эту:
const handleGoogleSignIn = async () => {
  try {
    await signInWithRedirect(auth, googleProvider); // <-- вместо signInWithPopup
    // После этого браузер уйдёт на Google, потом вернётся и useEffect выше обработает
  } catch (error) {
    console.error("Ошибка signInWithRedirect:", error);
  }
};

});
