using MedicalAssistant.Services_Abstraction.Contracts;
using MedicalAssistant.Shared.DTOs.AppointmentsDTOs;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;

namespace MedicalAssistant.Presentation.Controllers
{
    [ApiController]
    [Route("api/appointments")]
    [Authorize]
    public class AppointmentsController : ControllerBase
    {
        private readonly IAppointmentService _appointmentService;

        public AppointmentsController(IAppointmentService appointmentService)
        {
            _appointmentService = appointmentService;
        }

        // POST /appointments
        [HttpPost]
        [ProducesResponseType(typeof(AppointmentDto), StatusCodes.Status201Created)]
        [ProducesResponseType(StatusCodes.Status400BadRequest)]
        public async Task<IActionResult> Create([FromBody] CreateAppointmentDto dto)
        {
            if (!ModelState.IsValid)
                return BadRequest(ModelState);

            // Get patientId from JWT token
            var patientIdClaim = User.FindFirst("PatientId")?.Value
                              ?? User.FindFirst("sub")?.Value;
            if (!int.TryParse(patientIdClaim, out var patientId))
                return Unauthorized(new { message = "Invalid token." });

            dto.PatientId = patientId;

            var appointment = await _appointmentService.CreateAppointmentAsync(dto);
            return CreatedAtAction(nameof(GetById), new { id = appointment.Id }, appointment);
        }

        // GET /appointments/my
        [HttpGet("my")]
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
            return Ok(appointment);
        }

        // PUT /appointments/{id}/confirm
        [HttpPut("{id}/confirm")]
        [ProducesResponseType(typeof(AppointmentDto), StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status404NotFound)]
        public async Task<IActionResult> Confirm(int id)
        {
            var updated = await UpdateStatusAsync(id, "Confirmed", null);
            if (updated == null) return NotFound(new { message = "Appointment not found." });
            return Ok(updated);
        }

        // PUT /appointments/{id}/cancel
        [HttpPut("{id}/pending")]
        [ProducesResponseType(typeof(AppointmentDto), StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status404NotFound)]
        public async Task<IActionResult> SetPending(int id)
        {
            var updated = await UpdateStatusAsync(id, "Pending", null);
            if (updated == null) return NotFound(new { message = "Appointment not found." });
            return Ok(updated);
        }

        // PUT /appointments/{id}/cancel
        [HttpPut("{id}/cancel")]
        [ProducesResponseType(typeof(AppointmentDto), StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status404NotFound)]
        public async Task<IActionResult> Cancel(int id, [FromBody] CancelAppointmentBody? body)
        {
            var updated = await UpdateStatusAsync(id, "Cancelled", body?.Reason);
            if (updated == null) return NotFound(new { message = "Appointment not found." });
            return Ok(updated);
        }

        // PUT /appointments/{id}/complete
        [HttpPut("{id}/complete")]
        [ProducesResponseType(typeof(AppointmentDto), StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status404NotFound)]
        public async Task<IActionResult> Complete(int id, [FromBody] CompleteAppointmentBody? body)
        {
            var updated = await UpdateStatusAsync(id, "Completed", body?.Notes);
            if (updated == null) return NotFound(new { message = "Appointment not found." });
            return Ok(updated);
        }

        // DELETE /appointments/{id}
        [HttpDelete("{id}")]
        [ProducesResponseType(StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status404NotFound)]
        public async Task<IActionResult> Delete(int id)
        {
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
    }

    public class CancelAppointmentBody
    {
        public string? Reason { get; set; }
    }

    public class CompleteAppointmentBody
    {
        public string? Notes { get; set; }
    }
}
