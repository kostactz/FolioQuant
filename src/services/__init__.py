"""
Application layer services for FolioQuant.

This module contains the business logic layer that coordinates between
the infrastructure layer (clients) and the domain layer (models).
"""

from .book_manager import BookManager

__all__ = ["BookManager"]
