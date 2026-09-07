"""The queue, for the people who work it.

Staff only. Everything here is about requests the agent declined to answer,
which is the other half of telling a customer that somebody is dealing with
it — the half that makes the sentence true.
"""

from fastapi import APIRouter, HTTPException, status

from app.api.deps import DBSessionDep, StaffUserDep
from app.schemas.cases import Case, Resolution
from app.services.cases import (
    CaseAlreadyClosedError,
    CaseAlreadyTakenError,
    CaseDesk,
    CaseNotFoundError,
)

router = APIRouter()


def _missing(reference: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"No open case with reference {reference}",
    )


@router.get("", response_model=list[Case])
async def waiting_cases(staff: StaffUserDep, db: DBSessionDep) -> list[Case]:
    """Every request still waiting for somebody, oldest first.

    Answered requests are not here. They are records rather than work, and a
    queue that lists them is a queue nobody trusts.
    """
    return [Case.model_validate(case) for case in await CaseDesk(db).waiting()]


@router.post("/{reference}/claim", response_model=Case)
async def claim_case(reference: str, staff: StaffUserDep, db: DBSessionDep) -> Case:
    """Take a case, if nobody else already has.

    A second person is refused rather than quietly replacing the first, which
    is the only version of this that stops two people doing the same work.
    """
    try:
        return Case.model_validate(await CaseDesk(db).claim(reference, staff.id))
    except CaseNotFoundError as exc:
        raise _missing(reference) from exc
    except CaseAlreadyTakenError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Case {reference} is already with somebody",
        ) from exc
    except CaseAlreadyClosedError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Case {reference} has already been resolved",
        ) from exc


@router.post("/{reference}/resolve", response_model=Case)
async def resolve_case(
    reference: str, done: Resolution, staff: StaffUserDep, db: DBSessionDep
) -> Case:
    """Close a case, recording who dealt with it and what they did.

    A second attempt is refused rather than accepted, because the later note
    would replace the account of whoever actually handled it.
    """
    try:
        return Case.model_validate(
            await CaseDesk(db).resolve(reference, staff.id, done.note)
        )
    except CaseNotFoundError as exc:
        raise _missing(reference) from exc
    except CaseAlreadyClosedError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Case {reference} has already been resolved",
        ) from exc
