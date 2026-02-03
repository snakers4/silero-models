# Запуск тестов

Окружение, необходимое для запуска тестов, сильно отличается от минимального окружения для инференса моделей синтеза. По большей части это связано с зависимостями библиотеки для расчёта UTMOS.

Запуск производится на **Python3.10**

Ноутбук нужно запускать в окружении с поставленными библиотеками из requirements_test.txt:
```bash
python3.10 -m venv venv_test
source venv_test/bin/activate
pip install -r requirements_test.txt
```

```
pip install ipykernel
python3.10 -m ipykernel install --user --name=venv_test --display-name="silero_test"
```
При запуске ноутбука в разделе kernel выбрать silero_test