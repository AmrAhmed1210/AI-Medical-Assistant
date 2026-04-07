using MedicalAssistant.Services_Abstraction.Contracts;
using MedicalAssistant.Shared.DTOs.AppointmentsDTOs;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;

namespace MedicalAssistant.Presentation.Controllers
{
    [ApiController]
    [Route("api/appointments")]
    public class AppointmentsController : ControllerBase
    {
        private readonly IAppointmentService _appointmentService;

        public AppointmentsController(IAppointmentService appointmentService)
        {
            _appointmentService = appointmentService;
        }

        // POST /api/appointments
        [HttpPost]
        [ProducesResponseType(typeof(AppointmentDto), StatusCodes.Status201Created)]
        [ProducesResponseType(StatusCodes.Status400BadRequest)]
        public async Task<IActionResult> Create([FromBody] CreateAppointmentDto dto)
        {
            if (!ModelState.IsValid)
                return BadRequest(ModelState);
            var appointment = await _appointmentService.CreateAppointmentAsync(dto);
            return CreatedAtAction(nameof(GetById), new { id = appointment.Id }, appointment);
        }

        // GET /api/appointments?status=Pending&page=1&pageSize=10
        [HttpGet]
        [ProducesResponseType(typeof(IEnumerable<AppointmentDto>), StatusCodes.Status200OK)]
        public async Task<IActionResult> GetAll([FromQuery] string? status, [FromQuery] int page = 1, [FromQuery] int pageSize = 20)
        {
            if (!string.IsNullOrEmpty(status))
            {
                // simple filter by status
                // for now return all for a status by fetching paginated and filtering
                var paged = await _appointmentService.GetPaginatedAppointmentsAsync(page, pageSize);
                var filtered = paged.Items.Where(a => string.Equals(a.Status, status, StringComparison.OrdinalIgnoreCase));
                return Ok(new { Items = filtered, paged.TotalCount, paged.PageNumber, paged.PageSize });
            }
            var all = await _appointmentService.GetPaginatedAppointmentsAsync(page, pageSize);
            return Ok(all);
        }

        // GET /api/appointments/{id}
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

        // PUT /api/appointments/{id}
        [HttpPut("{id}")]
        [ProducesResponseType(typeof(AppointmentDto), StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status400BadRequest)]
        [ProducesResponseType(StatusCodes.Status404NotFound)]
        public async Task<IActionResult> Update(int id, [FromBody] UpdateAppointmentDto dto)
        {
            if (!ModelState.IsValid)
                return BadRequest(ModelState);
            if (id != dto.Id)
                return BadRequest(new { message = "Id mismatch." });
            var updated = await _appointmentService.UpdateAppointmentAsync(dto);
            if (updated == null)
                return NotFound(new { message = "Appointment not found." });
            return Ok(updated);
        }

        // PUT /api/appointments/{id}/confirm
        [HttpPut("{id}/confirm")]
        [ProducesResponseType(typeof(AppointmentDto), StatusCodes.Status200OK)]
        public async Task<IActionResult> Confirm(int id)
        {
            var updated = await _appointmentService.ConfirmAppointmentAsync(id);
            if (updated == null) return NotFound(new { message = "Appointment not found." });
            return Ok(updated);
        }

        // PUT /api/appointments/{id}/cancel
        [HttpPut("{id}/cancel")]
        [ProducesResponseType(typeof(AppointmentDto), StatusCodes.Status200OK)]
        public async Task<IActionResult> Cancel(int id, [FromBody] CancelRequest req)
        {
            var updated = await _appointmentService.CancelAppointmentAsync(id, req.Reason);
            if (updated == null) return NotFound(new { message = "Appointment not found." });
            return Ok(updated);
        }

        // PUT /api/appointments/{id}/complete
        [HttpPut("{id}/complete")]
        [ProducesResponseType(typeof(AppointmentDto), StatusCodes.Status200OK)]
        public async Task<IActionResult> Complete(int id, [FromBody] CompleteRequest req)
        {
            var updated = await _appointmentService.CompleteAppointmentAsync(id, req.Notes);
            if (updated == null) return NotFound(new { message = "Appointment not found." });
            return Ok(updated);
        }

        // DELETE /api/appointments/{id}
        [HttpDelete("{id}")]
        [ProducesResponseType(StatusCodes.Status204NoContent)]
        [ProducesResponseType(StatusCodes.Status404NotFound)]
        public async Task<IActionResult> Delete(int id)
        {
            var deleted = await _appointmentService.DeleteAppointmentAsync(id);
            if (!deleted)
                return NotFound(new { message = "Appointment not found." });
            return NoContent();
        }

        public class CancelRequest { public string Reason { get; set; } = string.Empty; }
        public class CompleteRequest { public string? Notes { get; set; } }
    }
}
