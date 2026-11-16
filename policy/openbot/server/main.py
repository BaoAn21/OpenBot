from aiohttp import web

from .api import init_api
from .frontend import init_frontend
from .fwdbwd import init_fwdbwd
from .zeroconf import register

app = web.Application()


async def up(app: web.Application):
    print("Server is running, press Ctrl-C to exit...")


app.on_startup.append(register)
# Before init_api and init_frontend, so its / and POST /upload win over theirs.
app.on_startup.append(init_fwdbwd)
app.on_startup.append(init_api)
app.on_startup.append(init_frontend)
app.on_startup.append(up)
