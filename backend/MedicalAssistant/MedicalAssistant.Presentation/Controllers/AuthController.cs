using MedicalAssistant.Services_Abstraction.Contracts;
using MedicalAssistant.Shared.DTOs.Auth;
using Microsoft.AspNetCore.Mvc;

namespace MedicalAssistant.Presentation.Controllers;

[ApiController]
[Route("api/[controller]")]
public class AuthController : ControllerBase
{
    private readonly IAuthService _authService;

    public AuthController(IAuthService authService)
    {
        _authService = authService;
    }

    [HttpPost("login")]
    public async Task<IActionResult> Login([FromBody] LoginDto loginDto)
    {
        try
        {
            var result = await _authService.LoginAsync(loginDto);

            if (result == null)
            {
                return Unauthorized(new { message = "Invalid email or password" });
            }

            return Ok(result);
        }
        catch (Exception ex)
        {
            return BadRequest(new { 
                error = "Debug Mode", 
                detail = ex.Message 
            });
        }
    }


    [HttpPost("register-admin-internal")]
public async Task<IActionResult> RegisterAdmin([FromBody] RegisterDto registerDto)
{
    var result = await _authService.RegisterAdminAsync(registerDto);
    if (!result) return BadRequest("Failed to create admin");
    return Ok("Admin created successfully");
}
}