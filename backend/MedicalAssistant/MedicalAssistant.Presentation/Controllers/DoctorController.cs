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

    // --- Public Endpoints (For Patients/Guests) ---

    [HttpGet]
    public async Task<ActionResult<IEnumerable<DoctorDTO>>> GetAll([FromQuery] int? specialtyId)
    {
        // التعديل هنا: البحث أصبح بـ specialtyId (int) وليس نصاً
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

    // --- Private Endpoints (Doctor Only) ---

    [Authorize(Roles = "Doctor")]
    [HttpGet("dashboard")]
    public async Task<ActionResult<DoctorDashboardDto>> GetDashboard()
    {
        var doctorId = GetCurrentUserId();
        return Ok(await _doctorService.GetDoctorDashboardAsync(doctorId));
    }

    [Authorize(Roles = "Doctor")]
    [HttpGet("profile")]
    public async Task<ActionResult<DoctorDetailsDTO>> GetProfile()
    {
        var doctorId = GetCurrentUserId();
        return Ok(await _doctorService.GetDoctorByIdAsync(doctorId));
    }

    [Authorize(Roles = "Doctor")]
    [HttpPut("profile")]
    public async Task<IActionResult> UpdateProfile([FromBody] DoctorUpdateDto dto)
    {
        var doctorId = GetCurrentUserId();
        await _doctorService.UpdateProfileAsync(doctorId, dto);
        return NoContent();
    }

    [Authorize(Roles = "Doctor")]
    [HttpPost("photo")]
    public async Task<IActionResult> UploadPhoto(IFormFile file)
    {
        var doctorId = GetCurrentUserId();
        var url = await _doctorService.UploadProfilePhotoAsync(doctorId, file);
        return Ok(new { photoUrl = url });
    }

    [Authorize(Roles = "Doctor")]
    [HttpGet("appointments")]
    public async Task<ActionResult<IEnumerable<AppointmentDto>>> GetDoctorAppointments([FromQuery] string? status)
    {
        var doctorId = GetCurrentUserId();
        return Ok(await _doctorService.GetAppointmentsByDoctorAsync(doctorId, status));
    }

    [Authorize(Roles = "Doctor")]
    [HttpGet("patients")]
    public async Task<ActionResult<IEnumerable<PatientDto>>> GetMyPatients([FromQuery] string? search)
    {
        var doctorId = GetCurrentUserId();
        return Ok(await _doctorService.GetPatientsByDoctorAsync(doctorId, search));
    }

    [Authorize(Roles = "Doctor")]
    [HttpGet("reports")]
    public async Task<ActionResult<IEnumerable<AIReportDto>>> GetAIReports([FromQuery] string? urgency, [FromQuery] int? patientId)
    {
        // تم جعل patientId (Nullable int) ليتوافق مع الـ Service
        var doctorId = GetCurrentUserId();
        return Ok(await _doctorService.GetAIReportsAsync(doctorId, urgency, patientId));
    }

    [Authorize(Roles = "Doctor")]
    [HttpGet("availability")]
    public async Task<ActionResult<IEnumerable<AvailabilityDto>>> GetAvailability()
    {
        var doctorId = GetCurrentUserId();
        return Ok(await _doctorService.GetAvailabilityAsync(doctorId));
    }

    [Authorize(Roles = "Doctor")]
    [HttpPut("availability")]
    public async Task<IActionResult> UpdateAvailability([FromBody] IEnumerable<AvailabilityDto> slots)
    {
        var doctorId = GetCurrentUserId();
        await _doctorService.UpdateAvailabilityAsync(doctorId, slots);
        return NoContent();
    }

    // Helper Method لجلب الـ ID من الـ Token
    private int GetCurrentUserId()
    {
        var claim = User.FindFirstValue(ClaimTypes.NameIdentifier);
        return string.IsNullOrEmpty(claim) ? 0 : int.Parse(claim);
    }
}