using MedicalAssistant.Services_Abstraction.Contracts;
using MedicalAssistant.Shared.DTOs.Secretary;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using System.Security.Claims;

namespace MedicalAssistant.Presentation.Controllers;

[ApiController]
[Route("api/[controller]")]
[Authorize]
public class SecretaryController : ControllerBase
{
    private readonly ISecretaryService _secretaryService;
    private readonly IDoctorService _doctorService;

    public SecretaryController(ISecretaryService secretaryService, IDoctorService doctorService)
    {
        _secretaryService = secretaryService;
        _doctorService = doctorService;
    }

    private int GetCurrentUserId()
    {
        var userIdStr = User.FindFirst(ClaimTypes.NameIdentifier)?.Value
                     ?? User.FindFirst("UserId")?.Value;
        return int.Parse(userIdStr ?? "0");
    }

    [HttpPost("add")]
    [Authorize(Roles = "Doctor,Admin")]
    public async Task<IActionResult> AddSecretary([FromBody] CreateSecretaryDto dto)
    {
        try
        {
            var result = await _secretaryService.AddSecretaryAsync(GetCurrentUserId(), dto);
            return Ok(result);
        }
        catch (Exception ex)
        {
            return BadRequest(new { message = ex.Message });
        }
    }

    [HttpGet("my-secretaries")]
    [Authorize(Roles = "Doctor,Admin")]
    public async Task<IActionResult> GetMySecretaries()
    {
        var result = await _secretaryService.GetSecretariesForDoctorAsync(GetCurrentUserId());
        return Ok(result);
    }

    [HttpDelete("{id}")]
    [Authorize(Roles = "Doctor,Admin")]
    public async Task<IActionResult> DeleteSecretary(int id)
    {
        var success = await _secretaryService.DeleteSecretaryAsync(GetCurrentUserId(), id);
        return success ? Ok() : NotFound();
    }

    // --- Secretary-specific endpoints ---

    [HttpGet("my-doctor")]
    [Authorize(Roles = "Secretary")]
    public async Task<IActionResult> GetMyDoctor()
    {
        var secretaryUserId = GetCurrentUserId();
        var doctorId = await _secretaryService.GetDoctorIdForSecretaryAsync(secretaryUserId);
        if (!doctorId.HasValue)
            return NotFound(new { message = "No doctor assigned." });

        var profile = await _doctorService.GetProfileAsync(doctorId.Value);
        if (profile == null)
            return NotFound(new { message = "Doctor not found." });

        return Ok(profile);
    }

    [HttpGet("my-doctor/patients")]
    [Authorize(Roles = "Secretary")]
    public async Task<IActionResult> GetMyDoctorPatients([FromQuery] string? search = null)
    {
        var secretaryUserId = GetCurrentUserId();
        var doctorId = await _secretaryService.GetDoctorIdForSecretaryAsync(secretaryUserId);
        if (!doctorId.HasValue)
            return NotFound(new { message = "No doctor assigned." });

        var patients = await _doctorService.GetPatientsAsync(doctorId.Value, search);
        return Ok(patients);
    }

    [HttpGet("my-doctor/availability")]
    [Authorize(Roles = "Secretary")]
    public async Task<IActionResult> GetMyDoctorAvailability()
    {
        var secretaryUserId = GetCurrentUserId();
        var doctorId = await _secretaryService.GetDoctorIdForSecretaryAsync(secretaryUserId);
        if (!doctorId.HasValue)
            return NotFound(new { message = "No doctor assigned." });

        var availability = await _doctorService.GetAvailabilityAsync(doctorId.Value);
        return Ok(availability);
    }

    [HttpPut("my-doctor/availability")]
    [Authorize(Roles = "Secretary")]
    public async Task<IActionResult> UpdateMyDoctorAvailability([FromBody] List<AvailabilityDto> data)
    {
        var secretaryUserId = GetCurrentUserId();
        var doctorId = await _secretaryService.GetDoctorIdForSecretaryAsync(secretaryUserId);
        if (!doctorId.HasValue)
            return NotFound(new { message = "No doctor assigned." });

        await _doctorService.UpdateAvailabilityAsync(doctorId.Value, data);

        var schedule = await _doctorService.GetMyScheduleAsync(doctorId.Value);
        if (schedule != null)
        {
            // Note: notification service not injected here, skip or inject if needed
        }

        return NoContent();
    }
}
