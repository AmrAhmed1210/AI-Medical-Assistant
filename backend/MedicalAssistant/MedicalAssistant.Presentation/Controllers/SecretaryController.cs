using MedicalAssistant.Services_Abstraction.Contracts;
using MedicalAssistant.Shared.DTOs.Secretary;
using MedicalAssistant.Shared.DTOs.DoctorDTOs;
using MedicalAssistant.Shared.DTOs.PatientDTOs;
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
    private readonly IPatientService _patientService;

    public SecretaryController(ISecretaryService secretaryService, IDoctorService doctorService, IPatientService patientService)
    {
        _secretaryService = secretaryService;
        _doctorService = doctorService;
        _patientService = patientService;
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
        return NoContent();
    }

    [HttpGet("my-doctor/schedule")]
    [Authorize(Roles = "Secretary")]
    public async Task<IActionResult> GetMyDoctorSchedule()
    {
        var secretaryUserId = GetCurrentUserId();
        var doctorId = await _secretaryService.GetDoctorIdForSecretaryAsync(secretaryUserId);
        if (!doctorId.HasValue)
            return NotFound(new { message = "No doctor assigned." });

        var schedule = await _doctorService.GetScheduleAsync(doctorId.Value);
        if (schedule == null)
            return NotFound(new { message = "Schedule not found." });

        return Ok(schedule);
    }

    [HttpPut("my-doctor/schedule-visibility")]
    [Authorize(Roles = "Secretary")]
    public async Task<IActionResult> UpdateMyDoctorScheduleVisibility([FromBody] bool isVisible)
    {
        var secretaryUserId = GetCurrentUserId();
        var doctorId = await _secretaryService.GetDoctorIdForSecretaryAsync(secretaryUserId);
        if (!doctorId.HasValue)
            return NotFound(new { message = "No doctor assigned." });

        var profile = await _doctorService.GetProfileAsync(doctorId.Value);
        if (profile?.UserId.HasValue != true)
            return NotFound(new { message = "Doctor user not found." });

        await _doctorService.UpdateScheduleVisibilityAsync(profile.UserId.Value, isVisible);
        return NoContent();
    }

    [HttpPost("create-walkin-patient")]
    [Authorize(Roles = "Secretary")]
    public async Task<IActionResult> CreateWalkInPatient([FromBody] CreateWalkInPatientDto dto)
    {
        if (!ModelState.IsValid)
            return BadRequest(ModelState);

        try
        {
            var email = string.IsNullOrWhiteSpace(dto.Email) 
                ? $"walkin_{Guid.NewGuid().ToString("N").Substring(0, 8)}@clinic.local"
                : dto.Email;

            var createDto = new CreatePatientDto
            {
                FullName = dto.FullName,
                Email = email,
                PhoneNumber = dto.PhoneNumber,
                DateOfBirth = DateTime.Now.AddYears(-30),
                Gender = "Male",
                Address = null,
                ImageUrl = null,
                BloodType = null,
                MedicalNotes = "Walk-in patient created by secretary",
            };

            var patient = await _patientService.CreatePatientAsync(createDto);
            return Ok(new { id = patient.Id, fullName = patient.FullName, email = patient.Email });
        }
        catch (InvalidOperationException ex)
        {
            return BadRequest(new { message = ex.Message });
        }
        catch (Exception ex)
        {
            return StatusCode(500, new { message = "Failed to create patient", details = ex.Message });
        }
    }
}
