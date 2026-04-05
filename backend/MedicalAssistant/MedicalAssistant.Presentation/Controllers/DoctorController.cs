using MedicalAssistant.Services_Abstraction.Contracts;
using MedicalAssistant.Shared.DTOs.AppointmentsDTOs;
using MedicalAssistant.Shared.DTOs.DoctorDTOs;
using MedicalAssistant.Shared.DTOs.PatientDTOs;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using System.Security.Claims;

namespace MedicalAssistant.Presentation.Controllers;

[ApiController]
[Route("api/doctors")]
public class DoctorsController : ControllerBase
{
    private readonly IDoctorService _doctorService;

    public DoctorsController(IDoctorService doctorService)
    {
        _doctorService = doctorService;
    }

    [HttpGet]
    public async Task<ActionResult<IEnumerable<DoctorDTO>>> GetAll([FromQuery] int? specialtyId)
    {
        var doctors = specialtyId.HasValue
            ? await _doctorService.GetDoctorsBySpecialtyAsync(specialtyId.Value)
            : await _doctorService.GetAllDoctorsAsync();
        return Ok(doctors);
    }

    [HttpGet("{id}")]
    public async Task<ActionResult<DoctorDetailsDTO>> GetById(int id)
    {
        var doctor = await _doctorService.GetDoctorByIdAsync(id);
        return doctor == null ? NotFound() : Ok(doctor);
    }

    [Authorize(Roles = "Doctor")]
    [HttpGet("dashboard")]
    public async Task<ActionResult<DoctorDashboardDto>> GetDashboard()
    {
        var doctorId = int.Parse(User.FindFirstValue(ClaimTypes.NameIdentifier)!);
        return Ok(await _doctorService.GetDoctorDashboardAsync(doctorId));
    }

    [Authorize(Roles = "Doctor")]
    [HttpGet("profile")]
    public async Task<ActionResult<DoctorDetailsDTO>> GetProfile()
    {
        var doctorId = int.Parse(User.FindFirstValue(ClaimTypes.NameIdentifier)!);
        return Ok(await _doctorService.GetDoctorByIdAsync(doctorId));
    }

    [Authorize(Roles = "Doctor")]
    [HttpPut("availability")]
    public async Task<IActionResult> UpdateAvailability(IEnumerable<AvailabilityDto> slots)
    {
        var doctorId = int.Parse(User.FindFirstValue(ClaimTypes.NameIdentifier)!);
        await _doctorService.UpdateAvailabilityAsync(doctorId, slots);
        return NoContent();
    }

    [HttpPut("profile")]
    public async Task<IActionResult> UpdateProfile(DoctorUpdateDto dto)
    {
        var doctorId = int.Parse(User.FindFirstValue(ClaimTypes.NameIdentifier)!);
        await _doctorService.UpdateProfileAsync(doctorId, dto);
        return NoContent();
    }

    [HttpPost("photo")]
    public async Task<IActionResult> UploadPhoto(IFormFile file)
    {
        var doctorId = int.Parse(User.FindFirstValue(ClaimTypes.NameIdentifier)!);
        var url = await _doctorService.UploadProfilePhotoAsync(doctorId, file);
        return Ok(new { photoUrl = url });
    }

    [HttpGet("appointments")]
    public async Task<ActionResult<IEnumerable<AppointmentDto>>> GetDoctorAppointments([FromQuery] string? status)
    {
        var doctorId = int.Parse(User.FindFirstValue(ClaimTypes.NameIdentifier)!);
        return Ok(await _doctorService.GetAppointmentsByDoctorAsync(doctorId, status));
    }

    [HttpGet("patients")]
    public async Task<ActionResult<IEnumerable<PatientDto>>> GetMyPatients([FromQuery] string? search)
    {
        var doctorId = int.Parse(User.FindFirstValue(ClaimTypes.NameIdentifier)!);
        return Ok(await _doctorService.GetPatientsByDoctorAsync(doctorId, search));
    }

    [HttpGet("reports")]
    public async Task<ActionResult<IEnumerable<AIReportDto>>> GetAIReports([FromQuery] string? urgency, [FromQuery] Guid? patientId)
    {
        var doctorId = int.Parse(User.FindFirstValue(ClaimTypes.NameIdentifier)!);
        return Ok(await _doctorService.GetAIReportsAsync(doctorId, urgency, patientId));
    }

    [HttpGet("availability")]
    public async Task<ActionResult<IEnumerable<AvailabilityDto>>> GetAvailability()
    {
        var doctorId = int.Parse(User.FindFirstValue(ClaimTypes.NameIdentifier)!);
        return Ok(await _doctorService.GetAvailabilityAsync(doctorId));
    }
}