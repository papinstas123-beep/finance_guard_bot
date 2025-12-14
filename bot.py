import os
import asyncio

from aiogram import Bot, Dispatcher, F, Router
from aiogram.types import (
    Message,
    CallbackQuery,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
)
from aiogram.fsm.state import StatesGroup, State
from aiogram.fsm.context import FSMContext
from aiogram.filters import Command
from dotenv import load_dotenv
from groq import Groq

load_dotenv()


# FSM States
class DeepAnalyze(StatesGroup):
    deep_income = State()
    deep_rent = State()
    deep_communal = State()
    deep_transport = State()
    deep_subs = State()
    deep_credits = State()
    deep_credit_sum = State()
    deep_processing = State()
    deep_result_short = State()
    deep_result_full = State()


class QuickAnalyze(StatesGroup):
    quick_income = State()
    quick_categories = State()
    quick_show = State()
    quick_recommendations = State()


class GoalStates(StatesGroup):
    goal_intro = State()
    goal_price = State()
    goal_term = State()
    goal_plan = State()


class CheckUploadStates(StatesGroup):
    waiting_for_file = State()


# Simple test router for check
test_router = Router()


@test_router.message(Command("ping"))
async def ping_command(msg: Message):
    await msg.answer("pong")


# --- Inline Keyboards ---
def main_menu_kb():
    """Главное меню бота"""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="⚡ Быстрый анализ", callback_data="quick_analyze")],
            [InlineKeyboardButton(text="🔍 Глубокий анализ", callback_data="deep_analyze")],
            [InlineKeyboardButton(text="🎯 Цель (накопить)", callback_data="goal_start")],
            [InlineKeyboardButton(text="📄 Загрузить чек/выписку", callback_data="upload_check")],
        ]
    )


def credits_kb():
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Да", callback_data="credits_yes")],
            [InlineKeyboardButton(text="Нет", callback_data="credits_no")],
        ]
    )


def result_menu_kb():
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Показать детальный отчёт", callback_data="show_deep_full")],
            [InlineKeyboardButton(text="Вернуться в меню", callback_data="return_to_menu")],
        ]
    )


def quick_result_kb():
    """Кнопки для быстрого анализа"""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Показать рекомендации", callback_data="quick_recommendations")],
            [InlineKeyboardButton(text="Глубокий анализ", callback_data="deep_analyze")],
            [InlineKeyboardButton(text="Вернуться в меню", callback_data="return_to_menu")],
        ]
    )


def quick_categories_kb():
    """Кнопки выбора категорий для быстрого анализа"""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="✅ Еда", callback_data="cat_food")],
            [InlineKeyboardButton(text="✅ Жильё", callback_data="cat_housing")],
            [InlineKeyboardButton(text="✅ Транспорт", callback_data="cat_transport")],
            [InlineKeyboardButton(text="✅ Подписки", callback_data="cat_subs")],
            [InlineKeyboardButton(text="✅ Покупки", callback_data="cat_shopping")],
            [InlineKeyboardButton(text="✅ Прочее", callback_data="cat_other")],
            [InlineKeyboardButton(text="➡️ Продолжить", callback_data="quick_categories_done")],
        ]
    )


# --- LLM API integration ---
async def get_llm_recommendations(user_data: dict, section: str = "deep"):
    """
    Формирует промпт и делает запрос к LLM API для выдачи персональных рекомендаций.
    section: deep | deep_full | quick | goal
    """
    system_prompt = """
Ты — строгий, уверенный финансовый наставник.

Твоё задание — анализировать личные финансы пользователя по его данным: доход, обязательные расходы (жильё, коммунальные платежи, транспорт, кредиты), подписки, ежедневные траты и финансовые цели.

Принципы работы:
Чёткий, прямой стиль, но без оскорблений.
Говори короткими абзацами и маркированными списками.
Всегда указывай порядок действий: шаг 1, шаг 2, шаг 3.
Делай акцент на том, где «утекают» деньги, что можно урезать без сильного падения качества жизни, как быстрее прийти к цели.
Не давай инвестиционных советов по конкретным акциям, фондам, криптовалютам. Работай только с бюджетом, расходами и накоплениями.
Если данных мало или они противоречат друг другу — задай 2–3 уточняющих вопроса, а потом всё равно выдай аккуратные рекомендации.

Всегда формируй ответ в структуре:
1. Краткая оценка ситуации.
2. Основные проблемы и «дыры» в бюджете.
3. Конкретные шаги экономии с примерными суммами/процентами.
4. План накоплений на цель (если цель указана).
5. Короткое жёсткое резюме-наставление.
"""

    text_subs = user_data.get("subs_raw") or user_data.get("deep_subs") or "-"
    text_goal = user_data.get("goal_text") or "-"
    quick_expenses = user_data.get("quick_expenses", "-")

    user_prompt = f"""
Данные пользователя для анализа бюджета:

Доход (чистыми в месяц): {user_data.get('income', '-')} ₽
Жильё (аренда/ипотека): {user_data.get('rent', '-')} ₽
Коммунальные платежи: {user_data.get('communal', '-')} ₽
Транспорт: {user_data.get('transport', '-')} ₽
Подписки (список и суммы, как есть):
{text_subs}
Ежемесячный платёж по кредитам: {user_data.get('credit_sum', '0')} ₽
Основные категории трат по ощущению пользователя: {quick_expenses}
Цель: {text_goal}
Сумма для накопления: {user_data.get('goal_sum', '-')} ₽
Желаемый срок: {user_data.get('goal_term', '-')} месяцев
"""

    # TODO: здесь должен быть реальный вызов LLM API с system_prompt и user_prompt
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
message = client.messages.create(            model="mixtral-8x7b-32768",
            max_tokens=2048,
system="Ты — ведущий независимый финансовый консультант с многолетним опытом. Твоя экспертиза: личные финансы, инвестирование, налоговое планирование, управление долгом, пенсионное планирование. Анализируя финансовую ситуацию пользователя, даёшь действенные, персонализированные рекомендации. Ты ВСЕГДА: 1) Приводишь КОНКРЕТНЫЕ цифры и расчёты, 2) Указываешь приоритеты (критичное → важное → желательное), 3) Даёшь пошаговый план с реальными сроками, 4) Объясняешь доступно, 5) Указываешь риски и возможности, 6) Адаптируешь под Россию (налоги, продукты, инфляция), 7) Используешь данные пользователя для расчётов, 8) Мотивируешь достижимыми целями, 9) Даёшь советы применяемые СЕГОДНЯ. Отвечай на русском, будь дружелюбен и уверен. Предупреждай о рисках инвестирования."            messages=[{"role": "user", "content": user_prompt}]
        )
        return message.content[0].text
    except Exception as e:
        return f"Ошибка Groq API: {str(e)}"

# --- Main Router ---
router = Router()


# --- Start Command and Main Menu ---
@router.message(Command("start"))
async def cmd_start(msg: Message, state: FSMContext):
    await state.clear()
    welcome_text = (
        "👋 Привет! Я FinancialGuardBot — твой строгий финансовый наставник.\n\n"
        "Я помогу тебе:\n"
        "• Проанализировать твои доходы и расходы\n"
        "• Найти, где утекают деньги\n"
        "• Составить план для достижения финансовых целей\n"
        "• Дать конкретные рекомендации по экономии\n\n"
        "Выбери, с чего начнём:"
    )
    await msg.answer(welcome_text, reply_markup=main_menu_kb())


@router.callback_query(F.data == "return_to_menu")
async def return_to_menu(call: CallbackQuery, state: FSMContext):
    await state.clear()
    await call.message.edit_text("Главное меню:", reply_markup=main_menu_kb())


# --- Quick Analyze Flow ---
@router.callback_query(F.data == "quick_analyze")
async def start_quick_analyze(call: CallbackQuery, state: FSMContext):
    await call.message.edit_reply_markup()
    await call.message.answer("⚡ Быстрый анализ\n\nТвой доход (чистыми в месяц)?")
    await state.set_state(QuickAnalyze.quick_income)


@router.message(QuickAnalyze.quick_income)
async def quick_income_step(msg: Message, state: FSMContext):
    try:
        income = int(msg.text.replace(" ", ""))
    except ValueError:
        await msg.reply("Введи сумму дохода числом, без лишних символов.")
        return

    await state.update_data(income=income, quick_categories_selected=[])
    await msg.answer(
        "Выбери категории трат, которые тебя беспокоят (можно несколько):",
        reply_markup=quick_categories_kb(),
    )
    await state.set_state(QuickAnalyze.quick_categories)


@router.callback_query(QuickAnalyze.quick_categories, F.data.startswith("cat_"))
async def quick_category_toggle(call: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    selected = data.get("quick_categories_selected", [])
    category = call.data.replace("cat_", "")
    category_names = {
        "food": "Еда",
        "housing": "Жильё",
        "transport": "Транспорт",
        "subs": "Подписки",
        "shopping": "Покупки",
        "other": "Прочее",
    }

    if category in selected:
        selected.remove(category)
        action = "❌"
    else:
        selected.append(category)
        action = "✅"

    await state.update_data(quick_categories_selected=selected)  # Обновляем текст кнопки

    kb = quick_categories_kb()
    for row in kb.inline_keyboard:
        for btn in row:
            if btn.callback_data == call.data:
                btn.text = f"{action} {category_names.get(category, category)}"
    await call.message.edit_reply_markup(reply_markup=kb)


@router.callback_query(QuickAnalyze.quick_categories, F.data == "quick_categories_done")
async def quick_categories_done(call: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    selected = data.get("quick_categories_selected", [])

    if not selected:
        await call.answer("Выбери хотя бы одну категорию!", show_alert=True)
        return

    category_names = {
        "food": "Еда",
        "housing": "Жильё",
        "transport": "Транспорт",
        "subs": "Подписки",
        "shopping": "Покупки",
        "other": "Прочее",
    }

    selected_names = [category_names.get(cat, cat) for cat in selected]
    quick_expenses = ", ".join(selected_names)
    await state.update_data(quick_expenses=quick_expenses)

    await call.message.edit_reply_markup()

    income = data.get("income", 0)
    await call.message.answer(
        f"📊 Промежуточный результат:\n\n"
        f"Доход: {income} ₽/мес\n"
        f"Категории трат: {quick_expenses}\n\n"
        f"Что дальше?",
        reply_markup=quick_result_kb(),
    )
    await state.set_state(QuickAnalyze.quick_show)


@router.callback_query(QuickAnalyze.quick_show, F.data == "quick_recommendations")
async def show_quick_recommendations(call: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    recommendation = await get_llm_recommendations(data, section="quick")

    await call.message.edit_reply_markup()
    await call.message.answer(
        f"💡 Рекомендации по быстрому анализу:\n\n{recommendation}",
        reply_markup=quick_result_kb(),
    )
    await state.set_state(QuickAnalyze.quick_recommendations)


# --- Deep Analyze Flow ---
@router.callback_query(F.data == "deep_analyze")
async def start_deep_analyze(call: CallbackQuery, state: FSMContext):
    await call.message.edit_reply_markup()
    await call.message.answer("🔍 Глубокий анализ\n\nТвой доход (чистыми в месяц)?")
    await state.set_state(DeepAnalyze.deep_income)


@router.message(DeepAnalyze.deep_income)
async def deep_income_step(msg: Message, state: FSMContext):
    try:
        income = int(msg.text.replace(" ", ""))
    except ValueError:
        await msg.reply("Введи сумму дохода числом, без лишних символов.")
        return

    await state.update_data(income=income)
    await msg.answer("Расходы на жильё (аренда/ипотека) в месяц?")
    await state.set_state(DeepAnalyze.deep_rent)


@router.message(DeepAnalyze.deep_rent)
async def deep_rent_step(msg: Message, state: FSMContext):
    try:
        rent = int(msg.text.replace(" ", ""))
    except ValueError:
        await msg.reply("Введи сумму аренды/ипотеки числом.")
        return

    await state.update_data(rent=rent)
    await msg.answer("Коммунальные платежи в месяц?")
    await state.set_state(DeepAnalyze.deep_communal)


@router.message(DeepAnalyze.deep_communal)
async def deep_communal_step(msg: Message, state: FSMContext):
    try:
        communal = int(msg.text.replace(" ", ""))
    except ValueError:
        await msg.reply("Введи сумму коммуналки числом.")
        return

    await state.update_data(communal=communal)
    await msg.answer("Расходы на транспорт в месяц?")
    await state.set_state(DeepAnalyze.deep_transport)


@router.message(DeepAnalyze.deep_transport)
async def deep_transport_step(msg: Message, state: FSMContext):
    try:
        transport = int(msg.text.replace(" ", ""))
    except ValueError:
        await msg.reply("Введи сумму транспорта числом.")
        return

    await state.update_data(transport=transport)
    await msg.answer(
        "Сколько уходит на платные подписки? (Введи список и ориентировочные суммы в несколько строк)"
    )
    await state.set_state(DeepAnalyze.deep_subs)


@router.message(DeepAnalyze.deep_subs)
async def deep_subs_step(msg: Message, state: FSMContext):
    user_subs_text = msg.text.strip()
    await state.update_data(subs_raw=user_subs_text)
    await msg.answer("Есть кредиты/рассрочки? Нажми «Да» или «Нет».", reply_markup=credits_kb())
    await state.set_state(DeepAnalyze.deep_credits)


@router.callback_query(DeepAnalyze.deep_credits, F.data == "credits_yes")
async def deep_has_credits(call: CallbackQuery, state: FSMContext):
    await call.message.edit_reply_markup()
    await call.message.answer("Ежемесячный платёж по кредитам?")
    await state.set_state(DeepAnalyze.deep_credit_sum)


@router.callback_query(DeepAnalyze.deep_credits, F.data == "credits_no")
async def deep_no_credits(call: CallbackQuery, state: FSMContext):
    await state.update_data(credit_sum=0)
    await call.message.edit_reply_markup()
    await call.message.answer("Анализирую данные…")
    await state.set_state(DeepAnalyze.deep_processing)
    await process_deep_analysis(call.message, state)


@router.message(DeepAnalyze.deep_credit_sum)
async def deep_credit_sum_step(msg: Message, state: FSMContext):
    try:
        credit_sum = int(msg.text.replace(" ", ""))
    except ValueError:
        await msg.reply("Введи ежемесячный платёж по кредитам числом.")
        return

    await state.update_data(credit_sum=credit_sum)
    await msg.answer("Анализирую данные…")
    await state.set_state(DeepAnalyze.deep_processing)
    await process_deep_analysis(msg, state)


async def process_deep_analysis(msg: Message, state: FSMContext):
    data = await state.get_data()
    recommendation = await get_llm_recommendations(data, section="deep")

    await msg.answer(
        f"📊 Краткий отчёт:\n\n{recommendation}",
        reply_markup=result_menu_kb(),
    )
    await state.set_state(DeepAnalyze.deep_result_short)


@router.callback_query(DeepAnalyze.deep_result_short, F.data == "show_deep_full")
async def show_deep_full(call: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    recommendation = await get_llm_recommendations(data, section="deep_full")

    await call.message.answer(
        f"📋 Детальный разбор:\n\n{recommendation}",
        reply_markup=result_menu_kb(),
    )
    await state.set_state(DeepAnalyze.deep_result_full)


# --- Goal Flow ---
@router.callback_query(F.data == "goal_start")
async def start_goal(call: CallbackQuery, state: FSMContext):
    await call.message.edit_reply_markup()
    await call.message.answer(
        "🎯 Цель (накопить)\n\n"
        "Опиши свою финансовую цель (например: отпуск, новый телефон, ремонт, машина):"
    )
    await state.set_state(GoalStates.goal_intro)


@router.message(GoalStates.goal_intro)
async def goal_intro_step(msg: Message, state: FSMContext):
    goal_text = msg.text.strip()
    if len(goal_text) < 2:
        await msg.reply("Опиши цель чуть подробнее.")
        return

    await state.update_data(goal_text=goal_text)
    await msg.answer("Какую сумму хочешь накопить на эту цель?")
    await state.set_state(GoalStates.goal_price)


@router.message(GoalStates.goal_price)
async def goal_price_step(msg: Message, state: FSMContext):
    try:
        goal_sum = int(msg.text.replace(" ", ""))
    except ValueError:
        await msg.reply("Введи сумму числом.")
        return

    await state.update_data(goal_sum=goal_sum)
    await msg.answer("На какой срок (в месяцах, или введи дату, например: '12' или '12.2025')?")
    await state.set_state(GoalStates.goal_term)


@router.message(GoalStates.goal_term)
async def goal_term_step(msg: Message, state: FSMContext):
    term = msg.text.strip()
    try:
        goal_term = int(term)
    except ValueError:
        goal_term = term

    await state.update_data(goal_term=goal_term)
    await msg.answer("Считаю план действий…")
    await state.set_state(GoalStates.goal_plan)
    await process_goal_plan(msg, state)


async def process_goal_plan(msg: Message, state: FSMContext):
    data = await state.get_data()
    recommendation = await get_llm_recommendations(data, section="goal")

    await msg.answer(
        f"🎯 План по достижению цели:\n\n{recommendation}",
        reply_markup=main_menu_kb(),
    )
    await state.clear()


# --- Check/Statement Upload ---
@router.callback_query(F.data == "upload_check")
async def start_check_upload(call: CallbackQuery, state: FSMContext):
    await call.message.edit_reply_markup()
    await call.message.answer(
        "📄 Загрузить чек/выписку\n\nПришли фото, документ или текст выписки по расходам."
    )
    await state.set_state(CheckUploadStates.waiting_for_file)


@router.message(F.text.lower().in_({"загрузить чек", "загрузить выписку"}))
async def start_check_upload_text(msg: Message, state: FSMContext):
    await state.set_state(CheckUploadStates.waiting_for_file)
    await msg.answer("Пришли фото, документ или текст выписки по расходам.")


@router.message(CheckUploadStates.waiting_for_file, F.document | F.photo | F.text)
async def on_file_received(msg: Message, state: FSMContext):
    file_type = "фото" if msg.photo else "документ" if msg.document else "текст"
    await msg.answer(
        f"✅ {file_type.capitalize()} получен. Обрабатываю...\n\n"
        "(В будущем здесь будет распознавание и автокатегоризация трат)"
    )
    await state.clear()
    await msg.answer("Главное меню:", reply_markup=main_menu_kb())


# --- Main Entry Point ---
async def main():
    bot_token = os.getenv("BOT_TOKEN")
    if not bot_token:
        raise ValueError(
            "BOT_TOKEN не найден в переменных окружения! Создайте файл .env с BOT_TOKEN=ваш_токен"
        )

    bot = Bot(token=bot_token)
    dp = Dispatcher()

    dp.include_router(test_router)
    dp.include_router(router)

    print("Бот запущен. Нажми Ctrl+C, чтобы остановить.")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())


