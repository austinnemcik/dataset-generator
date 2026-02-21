from fastapi.responses import JSONResponse

def valid_http(statusCode):
    if not isinstance(statusCode, int):
        return False
    return (statusCode > 100 and statusCode <= 599)

def response_builder(
        *,
        success: bool, 
        message: str, 
        count: int | None = None, 
        errors: int | None = None, 
        statusCode: int = 200):
    status = statusCode
    if not (valid_http(statusCode)):
        status = 500

    return JSONResponse({
        "success": success,
        "message": message,
        "amount": count,
        "errors": errors
    }, status_code=status)

