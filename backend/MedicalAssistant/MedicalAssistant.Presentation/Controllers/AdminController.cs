using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Authorization;
using MedicalAssistant.Services_Abstraction.Contracts;
using MedicalAssistant.Shared.DTOs.Admin;
using MedicalAssistant.Shared.DTOs.Common;

namespace MedicalAssistant.API.Controllers;

[Authorize(Roles = "Admin")]
[ApiController]
[Route("api/admin")]
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
        var result = await _adminService.GetSystemStatsAsync();
        return Ok(result);
    }

    [HttpGet("users")]
    public async Task<ActionResult<PagedResult<UserManagementDto>>> GetUsers(
        [FromQuery] string? role = null,
        [FromQuery] string? search = null,
        [FromQuery] int page = 1,
        [FromQuery] int pageSize = 20)
    {
        var result = await _adminService.GetUsersAsync(page, pageSize, search, role);
        return Ok(result);
    }

    [HttpPost("users")]
    public async Task<ActionResult<UserManagementDto>> CreateUser([FromBody] CreateUserRequest request)
    {
        if (request == null) return BadRequest();
        var user = await _adminService.CreateUserAsync(request);
        return Ok(user);
    }

    [HttpPut("users/{id}/toggle")]
    public async Task<IActionResult> ToggleUser(int id)
    {
        var success = await _adminService.ToggleUserStatusAsync(id);
        if (!success) return NotFound();
        return NoContent();
    }

    [HttpDelete("users/{id}")]
    public async Task<IActionResult> DeleteUser(int id)
    {
        var success = await _adminService.DeleteUserAsync(id);
        if (!success) return NotFound();
        return NoContent();
    }

    [HttpGet("models")]
    public async Task<ActionResult<IEnumerable<ModelVersionDto>>> ListModels()
    {
        var models = await _adminService.ListModelVersionsAsync();
        return Ok(models);
    }

    [HttpPost("reload-model")]
    public async Task<IActionResult> ReloadModel([FromBody] ReloadModelRequest request)
    {
        if (request == null || string.IsNullOrEmpty(request.AgentName))
            return BadRequest();

        await _adminService.ReloadAiModelAsync(request.AgentName);
        return Ok(new { message = "Model reload signal sent successfully" });
    }
}