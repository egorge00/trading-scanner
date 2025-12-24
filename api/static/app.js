const startButton = document.getElementById("startDemo");
const restartButton = document.getElementById("restartDemo");
const boardContainer = document.getElementById("boardContainer");
const message = document.getElementById("message");
const lichessLink = document.getElementById("lichessLink");

const START_ENDPOINT = "/start-game-demo";
const EMBED_THEME = "blue";
const EMBED_BG = "light";

const setMessage = (text) => {
  message.textContent = text;
};

const clearBoard = () => {
  boardContainer.innerHTML = "";
  lichessLink.classList.add("hidden");
  lichessLink.removeAttribute("href");
};

const buildEmbedUrl = (gameId) => {
  const params = new URLSearchParams({ theme: EMBED_THEME, bg: EMBED_BG });
  return `https://lichess.org/embed/${gameId}?${params.toString()}`;
};

const renderBoard = (gameId) => {
  const wrapper = document.createElement("div");
  wrapper.className = "board-wrapper";

  const iframe = document.createElement("iframe");
  iframe.src = buildEmbedUrl(gameId);
  iframe.title = "Partie Lichess";
  iframe.allow = "autoplay; fullscreen";

  wrapper.appendChild(iframe);
  boardContainer.appendChild(wrapper);

  lichessLink.href = `https://lichess.org/${gameId}`;
  lichessLink.classList.remove("hidden");
};

const updateControls = (running) => {
  startButton.disabled = running;
  restartButton.disabled = !running;
};

const startGame = async () => {
  setMessage("");
  clearBoard();
  updateControls(true);

  try {
    const response = await fetch(START_ENDPOINT, { method: "GET" });
    if (!response.ok) {
      throw new Error(`Erreur réseau (${response.status})`);
    }

    const payload = await response.json();
    const gameId = payload.game_id || payload.gameId || payload.id;
    if (!gameId) {
      throw new Error("Réponse invalide : game_id manquant.");
    }

    renderBoard(gameId);
  } catch (error) {
    updateControls(false);
    setMessage("Impossible de démarrer la démo. Veuillez réessayer.");
    console.error(error);
  }
};

startButton.addEventListener("click", () => {
  startGame();
});

restartButton.addEventListener("click", () => {
  startGame();
});
