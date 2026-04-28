using MedicalAssistant.Services_Abstraction.Contracts;
using MedicalAssistant.Domain.Contracts;
using MedicalAssistant.Domain.Entities.PatientModule;
using MedicalAssistant.Domain.Entities.UserModule;
using MedicalAssistant.Shared.DTOs.AppointmentsDTOs;
using MedicalAssistant.Shared.DTOs.DoctorDTOs;
using MedicalAssistant.Shared.DTOs.ReviewDTOs;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using System.Linq;

namespace MedicalAssistant.Presentation.Controllers
{
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
        private readonly ISessionService _sessionService;
        private readonly IMessageService _messageService;
        private readonly IUnitOfWork _unitOfWork;
        private readonly IPhotoService _photoService;

        public DoctorsController(
            IDoctorService doctorService,
            INotificationService notificationService,
            ISessionService sessionService,
            IMessageService messageService,
            IUnitOfWork unitOfWork,
            IPhotoService photoService)
        {
            _doctorService = doctorService;
            _notificationService = notificationService;
            _sessionService = sessionService;
            _messageService = messageService;
            _unitOfWork = unitOfWork;
            _photoService = photoService;
        }

        // GET: api/doctors
        // GET: api/doctors?specialtyId=1
        [HttpGet]
        public async Task<ActionResult<IEnumerable<DoctorDTO>>> GetAllDoctors(
            [FromQuery] int? specialtyId = null)
        {
            var doctors = specialtyId.HasValue
                ? await _doctorService.GetDoctorsBySpecialtyAsync(specialtyId.Value)
                : await _doctorService.GetAllDoctorsAsync();

            return Ok(doctors);
        }

        // GET: api/doctors/{id}
        [HttpGet("{id}")]
        public async Task<ActionResult<DoctorDetailsDTO>> GetDoctorById(int id)
        {
            var doctor = await _doctorService.GetDoctorByIdAsync(id);
            if (doctor == null)
                return NotFound(new { message = "Doctor not found." });
            return Ok(doctor);
        }

        // POST: api/doctors/apply
        [AllowAnonymous]
        [HttpPost("apply")]
        public async Task<IActionResult> ApplyForDoctorAccount([FromBody] ApplyDoctorRequest request)
        {
            if (request == null || string.IsNullOrWhiteSpace(request.Email))
                return BadRequest(new { message = "Invalid application data." });

            if (string.IsNullOrWhiteSpace(request.Name))
                return BadRequest(new { message = "Full name is required." });

            if (string.IsNullOrWhiteSpace(request.Phone))
                return BadRequest(new { message = "Phone number is required." });

            if (string.IsNullOrWhiteSpace(request.LicenseNumber))
                return BadRequest(new { message = "License / National ID is required." });

            try
            {
                await _doctorService.ApplyForDoctorAccountAsync(request);
                return Ok(new { message = "Your application has been received. We will contact you soon." });
            }
            catch (InvalidOperationException ex)
            {
                return Conflict(new { message = ex.Message });
            }
        }

        private int GetDoctorId()
        {
            var userIdClaim = User.Claims
                .FirstOrDefault(c => c.Type == "UserId")?.Value;
            return int.TryParse(userIdClaim, out var id) ? id : 0;
        }

        // GET: api/doctors/dashboard
        [Authorize(Roles = "Doctor")]
        [HttpGet("dashboard")]
        public async Task<ActionResult<DoctorDashboardDto>> GetDashboard()
        {
            var doctorId = GetDoctorId();
            if (doctorId <= 0) return Unauthorized(new { message = "Invalid token." });
            var result = await _doctorService.GetDashboardAsync(doctorId);
            return Ok(result);
        }

        // GET: api/doctors/profile
        [Authorize(Roles = "Doctor")]
        [HttpGet("profile")]
        public async Task<ActionResult<DoctorDetailDto>> GetProfile()
        {
            var doctorId = GetDoctorId();
            if (doctorId <= 0) return Unauthorized(new { message = "Invalid token." });
            var result = await _doctorService.GetProfileAsync(doctorId);
            if (result == null) return NotFound(new { message = "Doctor not found." });
            return Ok(result);
        }

        // PUT: api/doctors/profile
        [Authorize(Roles = "Doctor")]
        [HttpPut("profile")]
        public async Task<IActionResult> UpdateProfile([FromBody] UpdateDoctorProfileRequest request)
        {
            var doctorId = GetDoctorId();
            if (doctorId <= 0) return Unauthorized(new { message = "Invalid token." });
            await _doctorService.UpdateProfileAsync(doctorId, request);
            return NoContent();
        }

        // GET: api/doctors/appointments
        [Authorize(Roles = "Doctor")]
        [HttpGet("appointments")]
        public async Task<ActionResult<IEnumerable<AppointmentDto>>> GetAppointments([FromQuery] string? status = null)
        {
            var doctorId = GetDoctorId();
            if (doctorId <= 0) return Unauthorized(new { message = "Invalid token." });
            var result = await _doctorService.GetAppointmentsAsync(doctorId, status);
            return Ok(result);
        }

        // DELETE: api/doctors/appointments/history
        [Authorize(Roles = "Doctor")]
        [HttpDelete("appointments/history")]
        public async Task<IActionResult> ClearHistory()
        {
            var doctorId = GetDoctorId();
            if (doctorId <= 0) return Unauthorized(new { message = "Invalid token." });

            await _doctorService.ClearAppointmentHistoryAsync(doctorId);
            return Ok(new { message = "History cleared successfully" });
        }

        // GET: api/doctors/patients
        [Authorize(Roles = "Doctor")]
        [HttpGet("patients")]
        public async Task<ActionResult<IEnumerable<PatientSummaryDto>>> GetPatients([FromQuery] string? search = null)
        {
            var doctorId = GetDoctorId();
            if (doctorId <= 0) return Unauthorized(new { message = "Invalid token." });
            var result = await _doctorService.GetPatientsAsync(doctorId, search);
            return Ok(result);
        }

        // GET: api/doctors/reports
        [Authorize(Roles = "Doctor")]
        [HttpGet("reports")]
        public async Task<ActionResult<IEnumerable<AIReportDto>>> GetReports([FromQuery] string? urgency = null)
        {
            var doctorId = GetDoctorId();
            if (doctorId <= 0) return Unauthorized(new { message = "Invalid token." });
            var result = await _doctorService.GetReportsAsync(doctorId, urgency);
            return Ok(result);
        }

        // GET: api/doctors/{id}/availability
        [AllowAnonymous]
        [HttpGet("{id}/availability")]
        public async Task<ActionResult<DoctorScheduleDto>> GetDoctorAvailability(int id)
        {
            var schedule = await _doctorService.GetScheduleAsync(id);
            if (schedule == null)
                return NotFound(new { message = "Doctor not found or no schedule" });
            return Ok(schedule);
        }

        // POST: api/doctors/{id}/notify-schedule
        [HttpPost("{id}/notify-schedule")]
        [Authorize]
        public IActionResult SubscribeToSchedule(int id)
        {
            return Ok(new { message = "You will be notified when schedule is ready" });
        }

        // GET: api/doctors/availability
        [Authorize(Roles = "Doctor")]
        [HttpGet("availability")]
        public async Task<ActionResult<IEnumerable<AvailabilityDto>>> GetAvailability()
        {
            var doctorId = GetDoctorId();
            if (doctorId <= 0) return Unauthorized(new { message = "Invalid token." });
            var result = await _doctorService.GetAvailabilityAsync(doctorId);
            return Ok(result);
        }

        // PUT: api/doctors/availability
        [Authorize(Roles = "Doctor")]
        [HttpPut("availability")]
        public async Task<IActionResult> UpdateAvailability([FromBody] List<AvailabilityDto> data)
        {
            var doctorId = GetDoctorId();
            if (doctorId <= 0) return Unauthorized(new { message = "Invalid token." });

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

        // PUT: api/doctors/schedule-visibility
        [Authorize(Roles = "Doctor")]
        [HttpPut("schedule-visibility")]
        public async Task<IActionResult> UpdateScheduleVisibility([FromBody] bool isVisible)
        {
            var doctorId = GetDoctorId();
            if (doctorId <= 0) return Unauthorized(new { message = "Invalid token." });

            await _doctorService.UpdateScheduleVisibilityAsync(doctorId, isVisible);

            return NoContent();
        }

        // POST: api/doctors/photo
        [Authorize(Roles = "Doctor")]
        [HttpPost("photo")]
        public async Task<IActionResult> UploadPhoto(IFormFile file)
        {
            if (file == null || file.Length == 0)
            {
                return BadRequest(new { message = "Photo is required." });
            }

            var doctorUserId = GetDoctorId();
            if (doctorUserId <= 0) return Unauthorized(new { message = "Invalid token." });

            var url = await _photoService.UploadPhotoAsync(file);
            await _doctorService.UpdatePhotoAsync(doctorUserId, url);

            return Ok(new { photoUrl = url });
        }

        // POST: api/doctors/apply/upload-cv
        [AllowAnonymous]
        [HttpPost("apply/upload-cv")]
        public async Task<IActionResult> UploadCv(IFormFile file)
        {
            if (file == null || file.Length == 0) return BadRequest(new { message = "File is required" });
            var url = await _photoService.UploadFileAsync(file);
            return Ok(new { url });
        }

        // GET: api/doctors/reviews
        [Authorize(Roles = "Doctor")]
        [HttpGet("reviews")]
        public async Task<ActionResult<IEnumerable<ReviewDto>>> GetMyReviews()
        {
            var doctorId = GetDoctorId();
            if (doctorId <= 0) return Unauthorized(new { message = "Invalid token." });
            var reviews = await _doctorService.GetMyReviewsAsync(doctorId);
            return Ok(reviews);
        }

        // POST: api/doctors/message-patient
        [HttpPost("message-patient")]
        [Authorize(Roles = "Doctor")]
        public async Task<IActionResult> MessagePatient([FromBody] MessagePatientRequest req)
        {
            if (req == null || string.IsNullOrWhiteSpace(req.PatientEmail) || string.IsNullOrWhiteSpace(req.Message))
            {
                return BadRequest(new { message = "Patient email and message are required." });
            }

            var doctorUserId = GetDoctorId();
            if (doctorUserId <= 0) return Unauthorized(new { message = "Invalid token." });

            var doctor = await _doctorService.GetProfileAsync(doctorUserId);
            if (doctor == null) return Unauthorized(new { message = "Doctor not found." });

            var patientEmail = req.PatientEmail.Trim().ToLowerInvariant();
            var patient = await _unitOfWork.Patients.GetByEmailAsync(patientEmail);
            if (patient == null || !patient.IsActive)
            {
                return NotFound(new { message = "Patient not found." });
            }

            var userRecord = (await _unitOfWork.Repository<User>().FindAsync(u => u.Email == patientEmail)).FirstOrDefault();
            if (userRecord == null)
            {
                userRecord = new User
                {
                    FullName = patient.FullName,
                    Email = patient.Email,
                    PasswordHash = patient.PasswordHash,
                    Role = "Patient",
                    PhoneNumber = patient.PhoneNumber,
                    BirthDate = patient.DateOfBirth,
                    PhotoUrl = patient.ImageUrl,
                    CreatedAt = patient.CreatedAt,
                    IsActive = true
                };
                await _unitOfWork.Repository<User>().AddAsync(userRecord);
                await _unitOfWork.SaveChangesAsync();

                patient.UserId = userRecord.Id;
                _unitOfWork.Patients.Update(patient);
                await _unitOfWork.SaveChangesAsync();
            }

            var sessionTitle = $"chat|p:{userRecord.Id}|d:{doctor.Id}|";
            var sessions = await _sessionService.GetSessionsByUserIdAsync(userRecord.Id);
            var existing = sessions.FirstOrDefault(s =>
                string.Equals(s.Title, sessionTitle, StringComparison.OrdinalIgnoreCase));

            var session = existing ?? await _sessionService.CreateSessionAsync(userRecord.Id, sessionTitle, "DoctorChat");

            await _messageService.SendMessageAsync(session.Id, doctorUserId, "doctor", req.Message.Trim());
            await _notificationService.NotifyNewMessage(
                patient.Email,
                doctor.FullName ?? "Your Doctor",
                req.Message.Trim(),
                session.Id,
                doctor.Id);

            return Ok(new { message = "Message sent", sessionId = session.Id });
        }

        // POST: api/doctors/self-deactivate
        [Authorize(Roles = "Doctor")]
        [HttpPost("self-deactivate")]
        public async Task<IActionResult> SelfDeactivate()
        {
            var doctorId = GetDoctorId();
            if (doctorId <= 0) return Unauthorized(new { message = "Invalid token." });

            var success = await _doctorService.SelfDeactivateAsync(doctorId);
            if (!success) return BadRequest(new { message = "Failed to deactivate account." });

            return Ok(new { message = "Account deactivated successfully. Contact admin to reactivate." });
        }
    }

    public class MessagePatientRequest
    {
        public string PatientEmail { get; set; } = string.Empty;
        public string Message { get; set; } = string.Empty;
    }

    public class UploadDoctorPhotoRequest
    {
        public IFormFile? Photo { get; set; }
    }
}
