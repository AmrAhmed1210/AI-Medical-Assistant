using MedicalAssistant.Services_Abstraction.Contracts;
using MedicalAssistant.Shared.DTOs.AppointmentsDTOs;
using MedicalAssistant.Shared.DTOs.DoctorDTOs;
using MedicalAssistant.Shared.DTOs.ReviewDTOs;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using System.Linq;

namespace MedicalAssistant.Presentation.Controllers
{
    // DTO wrapper required by Swashbuckle when using [FromForm] with IFormFile
    public class UploadPhotoRequest
    {
        public IFormFile File { get; set; } = null!;
    }

    [ApiController]
    [Route("api/doctors")]   
    public class DoctorsController : ControllerBase
    {
        private readonly IDoctorService _doctorService;
        private readonly INotificationService _notificationService;

        public DoctorsController(
            IDoctorService doctorService,
            INotificationService notificationService)
        {
            _doctorService = doctorService;
            _notificationService = notificationService;
        }

        // GET /doctors
        // GET /doctors?specialtyId=1
        [HttpGet]
        public async Task<ActionResult<IEnumerable<DoctorDTO>>> GetAllDoctors(
            [FromQuery] int? specialtyId = null)
        {
            var doctors = specialtyId.HasValue
                ? await _doctorService.GetDoctorsBySpecialtyAsync(specialtyId.Value)
                : await _doctorService.GetAllDoctorsAsync();

            return Ok(doctors);
        }

        // GET /doctors/{id}
        [HttpGet("{id}")]
        public async Task<ActionResult<DoctorDetailsDTO>> GetDoctorById(int id)
        {
            var doctor = await _doctorService.GetDoctorByIdAsync(id);
            if (doctor == null)
                return NotFound(new { message = "Doctor not found." });
            return Ok(doctor);
        }

        private int GetDoctorId()
        {
            var userIdClaim = User.Claims
                .FirstOrDefault(c => c.Type == "UserId")?.Value;
            return int.TryParse(userIdClaim, out var id) ? id : 0;
        }

        [Authorize(Roles = "Doctor")]
        [HttpGet("dashboard")]
        public async Task<ActionResult<DoctorDashboardDto>> GetDashboard()
        {
            var doctorId = GetDoctorId();
            if (doctorId <= 0) return Unauthorized();
            var result = await _doctorService.GetDashboardAsync(doctorId);
            return Ok(result);
        }

        [Authorize(Roles = "Doctor")]
        [HttpGet("profile")]
        public async Task<ActionResult<DoctorDetailDto>> GetProfile()
        {
            var doctorId = GetDoctorId();
            if (doctorId <= 0) return Unauthorized();
            var result = await _doctorService.GetProfileAsync(doctorId);
            if (result == null) return NotFound();
            return Ok(result);
        }

        [Authorize(Roles = "Doctor")]
        [HttpPut("profile")]
        public async Task<IActionResult> UpdateProfile([FromBody] UpdateDoctorProfileRequest request)
        {
            var doctorId = GetDoctorId();
            if (doctorId <= 0) return Unauthorized();
            await _doctorService.UpdateProfileAsync(doctorId, request);
            return NoContent();
        }

        [Authorize(Roles = "Doctor")]
        [HttpGet("appointments")]
        public async Task<ActionResult<IEnumerable<AppointmentDto>>> GetAppointments([FromQuery] string? status = null)
        {
            var doctorId = GetDoctorId();
            if (doctorId <= 0) return Unauthorized();
            var result = await _doctorService.GetAppointmentsAsync(doctorId, status);
            return Ok(result);
        }

        [Authorize(Roles = "Doctor")]
        [HttpGet("patients")]
        public async Task<ActionResult<IEnumerable<PatientSummaryDto>>> GetPatients([FromQuery] string? search = null)
        {
            var doctorId = GetDoctorId();
            if (doctorId <= 0) return Unauthorized();
            var result = await _doctorService.GetPatientsAsync(doctorId, search);
            return Ok(result);
        }

        [Authorize(Roles = "Doctor")]
        [HttpGet("reports")]
        public async Task<ActionResult<IEnumerable<AIReportDto>>> GetReports([FromQuery] string? urgency = null)
        {
            var doctorId = GetDoctorId();
            if (doctorId <= 0) return Unauthorized();
            var result = await _doctorService.GetReportsAsync(doctorId, urgency);
            return Ok(result);
        }

        [AllowAnonymous]
        [HttpGet("{id}/availability")]
        public async Task<ActionResult<DoctorScheduleDto>> GetDoctorAvailability(int id)
        {
            var schedule = await _doctorService.GetScheduleAsync(id);
            if (schedule == null)
                return NotFound(new { message = "Doctor not found or no schedule" });
            return Ok(schedule);
        }

        [HttpPost("{id}/notify-schedule")]
        [Authorize]
        public async Task<IActionResult> SubscribeToSchedule(int id)
        {
            // Simple: store in a table or just return OK for now
            return Ok(new { message = "You will be notified when schedule is ready" });
        }

        [Authorize(Roles = "Doctor")]
        [HttpGet("availability")]
        public async Task<ActionResult<IEnumerable<AvailabilityDto>>> GetAvailability()
        {
            var doctorId = GetDoctorId();
            if (doctorId <= 0) return Unauthorized();
            var result = await _doctorService.GetAvailabilityAsync(doctorId);
            return Ok(result);
        }

        [Authorize(Roles = "Doctor")]
        [HttpPut("availability")]
        public async Task<IActionResult> UpdateAvailability([FromBody] List<AvailabilityDto> data)
        {
            var doctorId = GetDoctorId();
            if (doctorId <= 0) return Unauthorized();
            
            await _doctorService.UpdateAvailabilityAsync(doctorId, data);
            
            var schedule = await _doctorService.GetMyScheduleAsync(doctorId);
            if (schedule != null)
            {
                await _notificationService.NotifyScheduleUpdated(
                    schedule.DoctorId,
                    schedule.DoctorName,
                    schedule.IsMobileEnabled);
            }
            
            return NoContent();
        }

        [Authorize(Roles = "Doctor")]
        [HttpPut("schedule-visibility")]
        public async Task<IActionResult> UpdateScheduleVisibility([FromBody] bool isVisible)
        {
            var doctorId = GetDoctorId();
            if (doctorId <= 0) return Unauthorized();
            
            await _doctorService.UpdateScheduleVisibilityAsync(doctorId, isVisible);
            
            return NoContent();
        }

        [Authorize(Roles = "Doctor")]
        [HttpPost("upload-photo")]
        [Consumes("multipart/form-data")]
        public async Task<IActionResult> UploadPhoto([FromForm] UploadPhotoRequest request)
        {
            if (request.File == null || request.File.Length == 0)
                return BadRequest("No file uploaded.");

            return Ok();
        }

        [Authorize(Roles = "Doctor")]
        [HttpGet("reviews")]
        public async Task<ActionResult<IEnumerable<ReviewDto>>> GetMyReviews()
        {
            var doctorId = GetDoctorId();
            if (doctorId <= 0) return Unauthorized();
            var reviews = await _doctorService.GetMyReviewsAsync(doctorId);
            return Ok(reviews);
        }

        // Self-deactivate account (Doctor can deactivate their own account)
        [Authorize(Roles = "Doctor")]
        [HttpPost("self-deactivate")]
        public async Task<IActionResult> SelfDeactivate()
        {
            var doctorId = GetDoctorId();
            if (doctorId <= 0) return Unauthorized();

            var success = await _doctorService.SelfDeactivateAsync(doctorId);
            if (!success) return BadRequest(new { message = "Failed to deactivate account." });

            return Ok(new { message = "Account deactivated successfully. Contact admin to reactivate." });
        }
    }
}
