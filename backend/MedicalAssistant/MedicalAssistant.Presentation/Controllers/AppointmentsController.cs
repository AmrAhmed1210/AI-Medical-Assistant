using MedicalAssistant.Services_Abstraction.Contracts;
using MedicalAssistant.Shared.DTOs.AppointmentsDTOs;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;

namespace MedicalAssistant.Presentation.Controllers
{
    [ApiController]
    [Route("appointments")]
    [Authorize]  // كل الـ endpoints تحتاج JWT
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
    }
}
