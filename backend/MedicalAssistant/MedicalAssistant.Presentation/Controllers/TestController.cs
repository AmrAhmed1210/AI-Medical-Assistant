using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using System.Security.Claims;
using MedicalAssistant.Domain.Contracts;
using MedicalAssistant.Domain.Entities.DoctorsModule;
using MedicalAssistant.Domain.Entities.AppointmentsModule;

namespace MedicalAssistant.Presentation.Controllers;

[ApiController]
[Route("api/test")]
[Authorize] // MUST be authorized to access JWT claims
public class TestController : ControllerBase
{
    private readonly IUnitOfWork _unitOfWork;

    public TestController(IUnitOfWork unitOfWork)
    {
        _unitOfWork = unitOfWork;
    }

    [HttpGet("availabilities")]
    [AllowAnonymous]
    public async Task<IActionResult> GetAvailabilities()
    {
        var doctors = (await _unitOfWork.Repository<Doctor>().GetAllAsync()).ToList();
        var availabilities = (await _unitOfWork.Repository<DoctorAvailability>().GetAllAsync()).ToList();
        var appointments = (await _unitOfWork.Repository<Appointment>().GetAllAsync()).ToList();

        return Ok(new
        {
            DoctorsCount = doctors.Count,
            Doctors = doctors.Select(d => new { d.Id, d.Name, d.UserId, d.IsAvailable, d.IsScheduleVisible }),
            AvailabilitiesCount = availabilities.Count,
            Availabilities = availabilities.Select(a => new { a.Id, a.DoctorId, a.DayOfWeek, a.StartTime, a.EndTime, a.IsAvailable }),
            AppointmentsCount = appointments.Count,
            Appointments = appointments.Select(a => new { a.Id, a.DoctorId, a.Date, a.Time, a.Status })
        });
    }

    [HttpGet("patient-summary")]
    public IActionResult GetPatientSummary()
    {
        // 🔒 SECURITY RULE APPLIED:
        // PatientId is extracted strictly from the JWT Claims.
        // It is PROHIBITED to accept [FromQuery] int patientId from the client.
        var claim = User.FindFirst("PatientId")?.Value;
        
        if (!int.TryParse(claim, out var patientId) || patientId <= 0)
        {
            return Unauthorized(new { error = "Unauthorized access or missing PatientId in token." });
        }

        // Mock data response based on the securely extracted patientId
        var mockData = new
        {
            Message = "This is a secure mock response.",
            PatientId = patientId, 
            Age = 35,
            Name = "Fake Patient Name" // As requested for the mock
        };

        return Ok(mockData);
    }
}
