using MedicalAssistant.Services_Abstraction.Contracts;
using MedicalAssistant.Shared.DTOs.DoctorDTOs;
using Microsoft.AspNetCore.Mvc;

namespace MedicalAssistant.Presentation.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class DoctorsController : ControllerBase
    {
        private readonly IDoctorService _doctorService;

        public DoctorsController(IDoctorService doctorService)
        {
            _doctorService = doctorService;
        }

        /// <summary>
        /// GET /api/doctors
        /// جلب قائمة كل الأطباء مع إمكانية فلترة التخصص
        /// </summary>
        [HttpGet]
        public async Task<ActionResult<IEnumerable<DoctorDTO>>> GetAllDoctors([FromQuery] int? specialtyId = null)
        {
            var doctors = specialtyId.HasValue
                ? await _doctorService.GetDoctorsBySpecialtyAsync(specialtyId.Value)
                : await _doctorService.GetAllDoctorsAsync();

            return Ok(doctors);
        }

        /// <summary>
        /// GET /api/doctors/{id}
        /// جلب تفاصيل طبيب واحد
        /// </summary>
        [HttpGet("{id}")]
        public async Task<ActionResult<DoctorDetailsDTO>> GetDoctorById(int id)
        {
            var doctor = await _doctorService.GetDoctorByIdAsync(id);

            if (doctor == null)
                return NotFound();

            return Ok(doctor);
        }
    }
}