"""Standalone worker entrypoint.

Run as: ``python -m apps.worker``.

This starts the vector store worker + cron in a dedicated process. Useful for
horizontal scale-out (multiple workers across machines).
"""
