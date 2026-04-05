import logging
import time
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

start = time.time()

# Устройство
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Устройство: {device}")

# Загружаем данные
log.info("Загружаем датасет Digits...")
digits = load_digits()
X, y = digits.data, digits.target
log.info(f"Загружено {len(X)} примеров, {X.shape[1]} признаков, {len(digits.target_names)} классов")

# Делим на train/test: 80% / 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
log.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

# Переводим в тензоры и отправляем на устройство
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test  = torch.tensor(X_test,  dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)
y_test  = torch.tensor(y_test,  dtype=torch.long).to(device)

# Модель: один линейный слой (логистическая регрессия)
model = nn.Linear(64, 10).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

# Обучение
max_iter = 1000
log.info(f"Обучаем (max_iter={max_iter})...")
for step in range(1, max_iter + 1):
    model.train()
    optimizer.zero_grad()
    loss = loss_fn(model(X_train), y_train)
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        log.info(f"  Шаг {step}/{max_iter} — loss: {loss.item():.4f}")

# Тест
model.eval()
with torch.no_grad():
    preds = model(X_test).argmax(dim=1)
    accuracy = (preds == y_test).float().mean().item()

log.info(f"Точность на тестовых данных: {accuracy * 100:.1f}%")
log.info(f"Время работы: {time.time() - start:.3f}с")
