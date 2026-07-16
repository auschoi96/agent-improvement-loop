from types import SimpleNamespace

from ail.jobs.validate_run_as import run_as_mismatches


def _job(name: str, *, sp: str | None = None, user: str | None = None, managed: bool = True):
    return SimpleNamespace(
        job_id=1,
        settings=SimpleNamespace(
            name=name,
            tags={"project": "agent-improvement-loop"} if managed else {},
            run_as=SimpleNamespace(service_principal_name=sp, user_name=user),
        ),
    )


def test_run_as_validation_accepts_only_expected_sp() -> None:
    jobs = [_job("ail-a", sp="sp-prod"), _job("unrelated", user="person", managed=False)]
    assert run_as_mismatches(jobs, "sp-prod") == []


def test_run_as_validation_names_human_and_wrong_sp_jobs() -> None:
    jobs = [_job("ail-human", user="person@example.com"), _job("ail-wrong", sp="sp-old")]
    mismatches = run_as_mismatches(jobs, "sp-prod")
    assert mismatches == [
        "ail-human: expected service principal sp-prod, found user:person@example.com",
        "ail-wrong: expected service principal sp-prod, found sp-old",
    ]


def test_run_as_validation_fails_when_no_managed_jobs_visible() -> None:
    assert run_as_mismatches([_job("unrelated", managed=False)], "sp-prod") == [
        "no managed agent-improvement-loop jobs were visible"
    ]
