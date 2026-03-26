using Microsoft.AspNetCore.Mvc;
using MedicalAssistant.Services_Abstraction.Contracts;
using MedicalAssistant.Shared.DTOs.Admin;

namespace MedicalAssistant.Presentation.Controllers;

[ApiController]
[Route("api/[controller]")]
public class AdminController : ControllerBase
{
    private readonly IAdminService _adminService;

    public AdminController(IAdminService adminService)
    {
        _adminService = adminService;
    }

    [HttpGet("stats")]
    public async Task<ActionResult<SystemStatsDto>> GetStats()
    {
        var stats = await _adminService.GetSystemStatsAsync();
        return Ok(stats);
    }

    [HttpGet("users")]
    public async Task<ActionResult<IEnumerable<UserManagementDto>>> GetUsers()
    {
        var users = await _adminService.GetAllUsersAsync();
        return Ok(users);
    }

    [HttpPost("users/{id}/toggle-status")]
    public async Task<IActionResult> ToggleStatus(int id)
    {
        var result = await _adminService.ToggleUserStatusAsync(id);
        if (!result) return NotFound();
        return Ok();
    }
}