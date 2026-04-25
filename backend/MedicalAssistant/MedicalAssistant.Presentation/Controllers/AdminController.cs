using Microsoft.AspNetCore.Mvc;
using MedicalAssistant.Shared.DTOs.DoctorDTOs;
using Microsoft.AspNetCore.Authorization;
using MedicalAssistant.Services_Abstraction.Contracts;
using MedicalAssistant.Shared.DTOs.Admin;
using MedicalAssistant.Shared.DTOs.Common;
using MedicalAssistant.Shared.DTOs.SessionDTOs;

namespace MedicalAssistant.API.Controllers;

[Authorize(Roles = "Admin")]
[ApiController]
[Route("api/admin")]
public class AdminController : ControllerBase
{
    private readonly IAdminService _adminService;
    private readonly ISessionService _sessionService;

    public AdminController(IAdminService adminService, ISessionService sessionService)
    {
        _adminService = adminService;
        _sessionService = sessionService;
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

    [HttpPatch("users/{id}/deactivate")]
    public async Task<IActionResult> DeactivateUser(int id, [FromQuery] string role)
    {
        if (string.IsNullOrWhiteSpace(role)) return BadRequest(new { message = "Role is required." });
        var success = await _adminService.DeactivateUserAsync(id, role);
        if (!success) return NotFound();
        return NoContent();
    }

    [HttpPatch("users/{id}/activate")]
    public async Task<IActionResult> ActivateUser(int id, [FromQuery] string role)
    {
        if (string.IsNullOrWhiteSpace(role)) return BadRequest(new { message = "Role is required." });
        var success = await _adminService.ActivateUserAsync(id, role);
        if (!success) return NotFound();
        return NoContent();
    }

    [HttpDelete("users/{id}")]
    public async Task<IActionResult> DeleteUser(int id, [FromQuery] string role)
    {
        if (string.IsNullOrWhiteSpace(role)) return BadRequest(new { message = "Role is required." });
        var success = await _adminService.DeleteUserAsync(id, role);
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

    [HttpGet("applications")]
    public async Task<ActionResult<IEnumerable<DoctorApplicationDto>>> GetApplications([FromQuery] string? status = null)
    {
        var apps = await _adminService.GetDoctorApplicationsAsync(status);
        return Ok(apps);
    }

    [HttpPost("applications/{id}/approve")]
    public async Task<IActionResult> ApproveApplication(int id)
    {
        var success = await _adminService.ApproveDoctorApplicationAsync(id);
        if (!success) return BadRequest(new { message = "Failed to approve application. It may already be processed." });
        return Ok(new { message = "Application approved. Doctor account created successfully." });
    }

    [HttpPost("applications/{id}/reject")]
    public async Task<IActionResult> RejectApplication(int id, [FromBody] RejectApplicationRequest? request = null)
    {
        var success = await _adminService.RejectDoctorApplicationAsync(id, request?.Reason);
        if (!success) return BadRequest(new { message = "Failed to reject application. It may already be processed." });
        return Ok(new { message = "Application rejected." });
    }

    [HttpGet("support-sessions")]
    public async Task<ActionResult<IEnumerable<SessionDto>>> GetSupportSessions()
    {
        var (sessions, _) = await _sessionService.GetPaginatedSessionsAsync(1, 1000);
        return Ok(sessions.Where(s => s.Type == "SupportChat"));
    }
}

public class RejectApplicationRequest
{
    public string? Reason { get; set; }
}
