import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_read_main(async_client: AsyncClient) -> None:
    response = await async_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


@pytest.mark.asyncio
async def test_the_documentation_wears_the_shops_own_mark(
    async_client: AsyncClient,
) -> None:
    """Both docs pages are served here rather than by the framework.

    Overriding them is the only way to set the icon on the tab, and an
    override is easy to break without anybody opening the page.
    """
    for page in ("/api/v1/docs", "/api/v1/redoc"):
        response = await async_client.get(page)
        assert response.status_code == 200, page
        assert "/assets/dorna-shop-mark.png" in response.text, page


@pytest.mark.asyncio
async def test_the_mark_and_the_logo_are_actually_served(
    async_client: AsyncClient,
) -> None:
    for asset in ("dorna-shop-mark.png", "dorna-shop-logo.png"):
        response = await async_client.get(f"/assets/{asset}")
        assert response.status_code == 200, asset
        assert response.headers["content-type"] == "image/png"


@pytest.mark.asyncio
async def test_the_schema_carries_the_logo_redoc_renders(
    async_client: AsyncClient,
) -> None:
    """Everything the docs display comes from the schema or the asset mount.

    Nothing branded reaches an API response: a health check returning a logo
    URL is decoration pretending to be data.
    """
    schema = (await async_client.get("/api/v1/openapi.json")).json()
    assert schema["info"]["x-logo"]["url"] == "/assets/dorna-shop-logo.png"
    assert "dorna-shop" not in (await async_client.get("/health")).text
