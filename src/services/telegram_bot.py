"""
telegram_bot.py — Telegram Interface
======================================
Connects the Orchestrator to Telegram so users can chat with
the match analysis system from their phone.

Commands:
    /start  — Welcome message
    /help   — List available commands
    /reset  — Clear conversation history

Any other text message gets routed to the Orchestrator,
which uses LLM + tool calling to handle it.
"""

import os
import logging
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from dotenv import load_dotenv

from src.agents.orchestrator.orchestrator import Orchestrator

load_dotenv()

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Global orchestrator instance — shared across all handlers
orchestrator = Orchestrator()


# =============================================================================
# COMMAND HANDLERS
# =============================================================================

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command."""
    await update.message.reply_text(
        "⚽ Football Match Analyzer\n\n"
        "I can analyze football match videos and answer questions about the results.\n\n"
        "Try:\n"
        '• "Analyze the match clip"\n'
        '• "Show me the match report"\n'
        '• "How did player 96 perform?"\n'
        '• "Compare the two teams"\n\n'
        "Commands:\n"
        "/help — Show this message\n"
        "/reset — Clear conversation history"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command."""
    await update.message.reply_text(
        "Available commands:\n"
        "/start — Welcome message\n"
        "/help — This help text\n"
        "/reset — Clear conversation history\n\n"
        "Or just type naturally:\n"
        '• "Analyze data/match_clip.mp4"\n'
        '• "What matches have been analyzed?"\n'
        '• "Show me the report"\n'
        '• "Stats for player 68"\n'
        '• "Compare the teams"'
    )


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /reset — clear conversation history but keep session store."""
    orchestrator.conversation_history = []
    await update.message.reply_text(
        "Conversation history cleared. Analyzed matches are still available."
    )


# =============================================================================
# MESSAGE HANDLER
# =============================================================================

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Route any text message to the Orchestrator.
    This is where the magic happens — user types naturally,
    Orchestrator uses LLM to understand and respond.
    """
    user_message = update.message.text
    user_name = update.message.from_user.first_name
    logger.info(f"[{user_name}] {user_message}")

    # Send "typing..." indicator while processing
    await update.message.chat.send_action("typing")

    try:
        response = orchestrator.chat(user_message)

        # Telegram has a 4096 character limit per message
        if len(response) > 4000:
            # Split into chunks
            chunks = [response[i:i+4000] for i in range(0, len(response), 4000)]
            for chunk in chunks:
                await update.message.reply_text(chunk)
        else:
            await update.message.reply_text(response)

    except Exception as e:
        logger.error(f"Error processing message: {e}")
        await update.message.reply_text(
            "Something went wrong. Try again, or use /reset to clear history."
        )


# =============================================================================
# BOT STARTUP
# =============================================================================

def run_bot():
    """Build and start the Telegram bot."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise ValueError("TELEGRAM_BOT_TOKEN not found in .env")

    app = ApplicationBuilder().token(token).build()

    # Register handlers
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("reset", reset_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot starting...")
    app.run_polling()


if __name__ == "__main__":
    run_bot()