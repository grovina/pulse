"""Local lab HTTP: static viewer plus a live student session."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .lab_export import LAB_DIR, build_graph
from .lab_sim import LabSession

app = FastAPI(title="Pulse lab")
SESSION = LabSession()


class ResetBody(BaseModel):
    protocol: str = "morning"


class EatBody(BaseModel):
    plate: str | None = "plate"
    carbs: float | None = None
    fats: float = 0.0
    proteins: float = 0.0


class WalkBody(BaseModel):
    minutes: int = 30
    intensity: float = Field(default=0.45, ge=0.0, le=1.0)


class AdvanceBody(BaseModel):
    minutes: int = Field(default=5, ge=1, le=180)


def _snap_or_409(fn):
    try:
        return fn()
    except ValueError as exc:
        if str(exc) == "asleep":
            raise HTTPException(409, "The body is asleep.") from exc
        if str(exc) == "empty meal":
            raise HTTPException(400, "That plate is empty.") from exc
        raise HTTPException(400, str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(404, f"Unknown {exc.args[0]}.") from exc


@app.get("/api/graph")
def api_graph():
    return build_graph()


@app.get("/api/protocols")
def api_protocols():
    return SESSION.protocols()


@app.get("/api/snapshot")
def api_snapshot():
    if not SESSION.run:
        SESSION.reset("morning")
    return SESSION.snapshot()


@app.post("/api/reset")
def api_reset(body: ResetBody):
    return _snap_or_409(lambda: SESSION.reset(body.protocol))


@app.post("/api/eat")
def api_eat(body: EatBody):
    if body.carbs is not None:
        return _snap_or_409(lambda: SESSION.eat(body.carbs, body.fats, body.proteins))
    return _snap_or_409(lambda: SESSION.eat_plate(body.plate or "plate"))


@app.post("/api/walk")
def api_walk(body: WalkBody):
    return _snap_or_409(lambda: SESSION.walk(body.minutes, body.intensity))


@app.post("/api/rest")
def api_rest():
    return _snap_or_409(SESSION.lie_down)


@app.post("/api/wake")
def api_wake():
    return _snap_or_409(SESSION.wake)


@app.post("/api/advance")
def api_advance(body: AdvanceBody):
    return _snap_or_409(lambda: SESSION.advance(body.minutes))


app.mount("/", StaticFiles(directory=str(LAB_DIR), html=True), name="lab")
