using Microsoft.AspNetCore.Mvc;
using MedicalAssistant.Services_Abstraction.Contracts;
using MedicalAssistant.Shared.DTOs.Admin;

namespace MedicalAssistant.API.Controllers;

[ApiController]
[Route("api/admin")]
public class AdminController : ControllerBase
{
    private readonly IAdminService _adminService;

    public AdminController(IAdminService adminService)
    {
        _adminService = adminService;
    }

    // ===============================
    // 📊 Stats
    // ===============================
    [HttpGet("stats")]
    public async Task<ActionResult<SystemStatsDto>> GetStats()
    {
        var result = await _adminService.GetSystemStatsAsync();
        return Ok(result);
    }

    // ===============================
    // 👥 Users
    // ===============================
    [HttpGet("users")]
    public async Task<IActionResult> GetUsers(
        [FromQuery] int page = 1,
        [FromQuery] int pageSize = 10,
        [FromQuery] string? search = null,
        [FromQuery] string? role = null)
    {
        try
        {
            var result = await _adminService.GetUsersAsync(page, pageSize, search, role);
            return Ok(result);
        }
        catch (Exception ex)
        {
            return StatusCode(500, new { message = "Server error", details = ex.Message });
        }
    }

    // ===============================
    // 🔄 Toggle
    // ===============================
    [HttpPut("users/{id}/toggle")]
    public async Task<IActionResult> ToggleUser(int id)
    {
        try
        {
            var success = await _adminService.ToggleUserStatusAsync(id);

            if (!success)
                return NotFound(new { message = "User not found" });

            return Ok(new { message = "User status updated" });
        }
        catch (Exception ex)
        {
            return StatusCode(500, new { message = "Toggle failed", details = ex.Message });
        }
    }

    // ===============================
    // ❌ Delete
    // ===============================
    [HttpDelete("users/{id}")]
    public async Task<IActionResult> DeleteUser(int id)
    {
        try
        {
            var success = await _adminService.DeleteUserAsync(id);

            if (!success)
                return NotFound(new { message = "User not found" });

            return Ok(new { message = "User deleted" });
        }
        catch (Exception ex)
        {
            return StatusCode(500, new { message = "Delete failed", details = ex.Message });
        }
    }

    // ===============================
    // ➕ Create
    // ===============================
    [HttpPost("users")]
    public async Task<IActionResult> CreateUser([FromBody] CreateUserRequest request)
    {
        try
        {
            if (request == null)
                return BadRequest(new { message = "Invalid request" });

            var user = await _adminService.CreateUserAsync(request);

            return Ok(user);
        }
        catch (Exception ex)
        {
            return StatusCode(500, new { message = "Create failed", details = ex.Message });
        }
    }
}