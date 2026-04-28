using MedicalAssistant.Shared.DTOs.DoctorDTOs;
namespace MedicalAssistant.Shared.DTOs.DoctorDTOs
{
    public class DoctorDetailsDTO : DoctorDTO
    {
        public int? Experience { get; set; }
        public string? Bio { get; set; }
        public DoctorScheduleDto? Schedule { get; set; }
    }
}
