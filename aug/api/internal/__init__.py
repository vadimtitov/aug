"""Internal HTTP surfaces — never mounted on the public app.

Modules here are separate ASGI applications bound to loopback.  Keeping them out
of ``aug/api/routers/`` is structural, not stylistic: everything in ``routers/``
is mounted by ``create_app()`` and therefore internet-reachable.
"""
