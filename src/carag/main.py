import uvicorn


def main():
    uvicorn.run(
        "carag.api.main:app",
        host="0.0.0.0",
        port=38779,
        log_level="info",
    )


if __name__ == "__main__":
    main()
