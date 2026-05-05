using MedicalAssistant.Services_Abstraction.Contracts;
using MedicalAssistant.Domain.Contracts;
using MedicalAssistant.Domain.Entities.DoctorsModule;
using MedicalAssistant.Shared.DTOs.AppointmentsDTOs;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using System.Security.Claims;

namespace MedicalAssistant.Presentation.Controllers
{
    [ApiController]
    [Route("api/appointments")]
    [Authorize]
    public class AppointmentsController : ControllerBase
    {
        private readonly IAppointmentService _appointmentService;
        private readonly IUnitOfWork _unitOfWork;

        public AppointmentsController(IAppointmentService appointmentService, IUnitOfWork unitOfWork)
        {
            _appointmentService = appointmentService;
            _unitOfWork = unitOfWork;
        }

        // POST /appointments
        [HttpPost]
        [Authorize(Roles = "Patient,Doctor,Secretary,Admin")]
        [ProducesResponseType(typeof(AppointmentDto), StatusCodes.Status201Created)]
        [ProducesResponseType(StatusCodes.Status400BadRequest)]
        public async Task<IActionResult> Create([FromBody] CreateAppointmentDto dto)
        {
            if (!ModelState.IsValid)
                return BadRequest(ModelState);

            var role = GetRole();
            if (string.Equals(role, "Patient", StringComparison.OrdinalIgnoreCase))
            {
                // Get patientId from JWT token
                var patientIdClaim = User.FindFirst("PatientId")?.Value
                                  ?? User.FindFirst("sub")?.Value;
                if (!int.TryParse(patientIdClaim, out var patientId))
                    return Unauthorized(new { message = "Invalid token." });

                dto.PatientId = patientId;
            }
            else if (dto.PatientId <= 0)
            {
                return BadRequest(new { message = "PatientId is required for non-patient bookings." });
            }

            // Reject bookings in the past
            if (!string.IsNullOrWhiteSpace(dto.Date) && !string.IsNullOrWhiteSpace(dto.Time))
            {
                if (DateTime.TryParse($"{dto.Date} {dto.Time}", out var slotUtc))
                {
                    // treat as local time (assume same timezone as server)
                    if (slotUtc < DateTime.Now.AddMinutes(-5))
                        return BadRequest(new { message = "Cannot book an appointment in the past." });
                }
            }

            var appointment = await _appointmentService.CreateAppointmentAsync(dto);
            return CreatedAtAction(nameof(GetById), new { id = appointment.Id }, appointment);
        }

        // GET /appointments/my
        [HttpGet("my")]
        [Authorize(Roles = "Patient")]
        [ProducesResponseType(typeof(IEnumerable<AppointmentDto>), StatusCodes.Status200OK)]
        public async Task<IActionResult> GetMyAppointments()
        {
            var patientIdClaim = User.FindFirst("PatientId")?.Value
                              ?? User.FindFirst("sub")?.Value;
            if (!int.TryParse(patientIdClaim, out var patientId))
                return Unauthorized(new { message = "Invalid token." });

            var appointments = await _appointmentService.GetAppointmentsByPatientIdAsync(patientId);
            return Ok(appointments);
        }

        // GET /appointments/{id}
        [HttpGet("{id}")]
        [ProducesResponseType(typeof(AppointmentDto), StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status404NotFound)]
        public async Task<IActionResult> GetById(int id)
        {
            var appointment = await _appointmentService.GetAppointmentByIdAsync(id);
            if (appointment == null)
                return NotFound(new { message = "Appointment not found." });

            if (!await CanAccessAppointmentAsync(appointment))
                return Forbid();

            return Ok(appointment);
        }

        // PUT /appointments/{id}/confirm
        [HttpPut("{id}/confirm")]
        [Authorize(Roles = "Doctor,Secretary,Admin")]
        [ProducesResponseType(typeof(AppointmentDto), StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status404NotFound)]
        public async Task<IActionResult> Confirm(int id)
        {
            var current = await _appointmentService.GetAppointmentByIdAsync(id);
            if (current == null) return NotFound(new { message = "Appointment not found." });
            if (!await CanAccessAppointmentAsync(current)) return Forbid();

            var updated = await UpdateStatusAsync(id, "Confirmed", null);
            if (updated == null) return NotFound(new { message = "Appointment not found." });
            return Ok(updated);
        }

        // PUT /appointments/{id}/cancel
        [HttpPut("{id}/pending")]
        [Authorize(Roles = "Doctor,Secretary,Admin")]
        [ProducesResponseType(typeof(AppointmentDto), StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status404NotFound)]
        public async Task<IActionResult> SetPending(int id)
        {
            var current = await _appointmentService.GetAppointmentByIdAsync(id);
            if (current == null) return NotFound(new { message = "Appointment not found." });
            if (!await CanAccessAppointmentAsync(current)) return Forbid();

            var updated = await UpdateStatusAsync(id, "Pending", null);
            if (updated == null) return NotFound(new { message = "Appointment not found." });
            return Ok(updated);
        }

        // PUT /appointments/{id}/cancel
        [HttpPut("{id}/cancel")]
        [Authorize(Roles = "Patient,Doctor,Secretary,Admin")]
        [ProducesResponseType(typeof(AppointmentDto), StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status404NotFound)]
        public async Task<IActionResult> Cancel(int id, [FromBody] CancelAppointmentBody? body)
        {
            var current = await _appointmentService.GetAppointmentByIdAsync(id);
            if (current == null) return NotFound(new { message = "Appointment not found." });
            if (!await CanAccessAppointmentAsync(current)) return Forbid();

            var updated = await UpdateStatusAsync(id, "Cancelled", body?.Reason);
            if (updated == null) return NotFound(new { message = "Appointment not found." });
            return Ok(updated);
        }

        // PUT /appointments/{id}/complete
        [HttpPut("{id}/complete")]
        [Authorize(Roles = "Doctor")]
        [ProducesResponseType(typeof(AppointmentDto), StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status404NotFound)]
        public async Task<IActionResult> Complete(int id, [FromBody] CompleteAppointmentBody? body)
        {
            var current = await _appointmentService.GetAppointmentByIdAsync(id);
            if (current == null) return NotFound(new { message = "Appointment not found." });
            if (!await CanDoctorAccessAppointmentAsync(current)) return Forbid();

            var updated = await UpdateStatusAsync(id, "Completed", body?.Notes);
            if (updated == null) return NotFound(new { message = "Appointment not found." });
            return Ok(updated);
        }

        // PUT /appointments/{id}/no-show
        [HttpPut("{id}/no-show")]
        [Authorize(Roles = "Doctor,Secretary,Admin")]
        [ProducesResponseType(typeof(AppointmentDto), StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status404NotFound)]
        public async Task<IActionResult> NoShow(int id)
        {
            var current = await _appointmentService.GetAppointmentByIdAsync(id);
            if (current == null) return NotFound(new { message = "Appointment not found." });
            if (!await CanAccessAppointmentAsync(current)) return Forbid();

            var updated = await UpdateStatusAsync(id, "NoShow", "Patient did not show up");
            if (updated == null) return NotFound(new { message = "Appointment not found." });
            return Ok(updated);
        }

        // PUT /appointments/{id}/reschedule
        [HttpPut("{id}/reschedule")]
        [Authorize(Roles = "Doctor,Secretary,Admin")]
        [ProducesResponseType(typeof(AppointmentDto), StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status404NotFound)]
        public async Task<IActionResult> Reschedule(int id, [FromBody] RescheduleAppointmentBody body)
        {
            var current = await _appointmentService.GetAppointmentByIdAsync(id);
            if (current == null) return NotFound(new { message = "Appointment not found." });
            if (!await CanAccessAppointmentAsync(current)) return Forbid();

            var dto = new UpdateAppointmentDto
            {
                Id = id,
                Status = "Rescheduled",
                Notes = body.Reason,
                PatientId = current.PatientId,
                DoctorId = current.DoctorId,
                AppointmentDate = DateTime.TryParse(body.NewDate, out var date) ? date : DateTime.UtcNow,
                AppointmentTime = TimeSpan.TryParse(body.NewTime, out var time) ? time : TimeSpan.Zero,
            };

            var updated = await _appointmentService.UpdateAppointmentAsync(dto);
            if (updated == null) return NotFound(new { message = "Appointment not found." });
            return Ok(updated);
        }

        // DELETE /appointments/{id}
        [HttpDelete("{id}")]
        [ProducesResponseType(StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status404NotFound)]
        public async Task<IActionResult> Delete(int id)
        {
            var current = await _appointmentService.GetAppointmentByIdAsync(id);
            if (current == null) return NotFound(new { message = "Appointment not found." });
            if (!await CanAccessAppointmentAsync(current)) return Forbid();

            var deleted = await _appointmentService.DeleteAppointmentAsync(id);
            if (!deleted)
                return NotFound(new { message = "Appointment not found." });

            // Frontend expects a message not 204
            return Ok(new { message = "Appointment cancelled successfully" });
        }

        // POST /appointments/{id}/rebook
        [HttpPost("{id}/rebook")]
        [ProducesResponseType(typeof(AppointmentDto), StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status404NotFound)]
        public async Task<IActionResult> Rebook(int id)
        {
            var patientIdClaim = User.FindFirst("PatientId")?.Value
                              ?? User.FindFirst("sub")?.Value;
            if (!int.TryParse(patientIdClaim, out var patientId))
                return Unauthorized(new { message = "Invalid token." });

            var appointment = await _appointmentService.RebookAppointmentAsync(id, patientId);
            if (appointment == null)
                return NotFound(new { message = "Appointment not found or cannot be rebooked." });

            return Ok(appointment);
        }

        private async Task<AppointmentDto?> UpdateStatusAsync(int id, string status, string? notes)
        {
            var current = await _appointmentService.GetAppointmentByIdAsync(id);
            if (current == null) return null;

            var dto = new UpdateAppointmentDto
            {
                Id = id,
                Status = status,
                Notes = notes,
                PatientId = current.PatientId,
                DoctorId = current.DoctorId,
                AppointmentDate = DateTime.UtcNow,
                AppointmentTime = TimeSpan.Zero
            };

            return await _appointmentService.UpdateAppointmentAsync(dto);
        }

        private string GetRole()
        {
            return User.FindFirstValue(ClaimTypes.Role)
                ?? User.FindFirstValue("role")
                ?? string.Empty;
        }

        private int? GetPatientId()
        {
            var patientIdClaim = User.FindFirst("PatientId")?.Value;
            return int.TryParse(patientIdClaim, out var patientId) ? patientId : null;
        }

        private int? GetUserId()
        {
            var userIdClaim = User.FindFirst(ClaimTypes.NameIdentifier)?.Value
                           ?? User.FindFirst("UserId")?.Value;
            return int.TryParse(userIdClaim, out var userId) ? userId : null;
        }

        private async Task<int?> GetDoctorProfileId()
        {
            var doctorIdClaim = User.FindFirst("DoctorId")?.Value;
            if (int.TryParse(doctorIdClaim, out var doctorId) && doctorId > 0)
                return doctorId;

            var userId = GetUserId();
            if (!userId.HasValue) return null;

            var doctor = (await _unitOfWork.Repository<Doctor>()
                .FindAsync(d => d.UserId == userId.Value))
                .FirstOrDefault();

            return doctor?.Id;
        }

        private async Task<bool> CanDoctorAccessAppointmentAsync(AppointmentDto appointment)
        {
            var doctorProfileId = await GetDoctorProfileId();
            if (!doctorProfileId.HasValue) return false;
            return appointment.DoctorId == doctorProfileId.Value;
        }

        [HttpGet("secretary")]
        [Authorize(Roles = "Secretary")]
        [ProducesResponseType(typeof(IEnumerable<AppointmentDto>), StatusCodes.Status200OK)]
        public async Task<IActionResult> GetSecretaryAppointments()
        {
            var userId = GetUserId();
            if (!userId.HasValue) return Unauthorized();

            var appointments = await _appointmentService.GetAppointmentsForSecretaryAsync(userId.Value);
            return Ok(appointments);
        }

        private async Task<bool> CanAccessAppointmentAsync(AppointmentDto appointment)
        {
            var role = GetRole();

            if (string.Equals(role, "Doctor", StringComparison.OrdinalIgnoreCase))
            {
                return await CanDoctorAccessAppointmentAsync(appointment);
            }

            if (string.Equals(role, "Secretary", StringComparison.OrdinalIgnoreCase))
            {
                var userId = GetUserId();
                if (!userId.HasValue) return false;

                var secretary = (await _unitOfWork.Repository<Secretary>().FindAsync(s => s.UserId == userId.Value)).FirstOrDefault();
                return secretary != null && appointment.DoctorId == secretary.DoctorId;
            }

            if (string.Equals(role, "Patient", StringComparison.OrdinalIgnoreCase))
            {
                var patientId = GetPatientId();
                return patientId.HasValue && appointment.PatientId == patientId.Value;
            }

            return false;
        }
    }

    public class CancelAppointmentBody
    {
        public string? Reason { get; set; }
    }

    public class CompleteAppointmentBody
    {
        public string? Notes { get; set; }
    }

    public class RescheduleAppointmentBody
    {
        public string NewDate { get; set; } = string.Empty;
        public string NewTime { get; set; } = string.Empty;
        public string? Reason { get; set; }
    }
}
